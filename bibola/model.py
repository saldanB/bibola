import torch
import pandas as pd
from bibola.MLP import MultiBatchesMLP
from bibola.spatial_autocorrelation import similarity_matrix, LIBSA
from bibola.knn import inter_batch_graph, intra_batch_graph
from bibola.topology import batch_similarity_preservation


class BaseIntegration():
    
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int,
                 optimizer_type: torch.optim.Optimizer = None, optimizer_kwargs: dict = None,
                 **kwargs):
        """
        Initialize the BaseIntegration.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (list): List of hidden channel sizes.
            out_channels (int): Number of output channels.
            optimizer_type (torch.optim.Optimizer): Type of optimizer for training.
            optimizer_kwargs (dict): Additional keyword arguments for the optimizer.
            **kwargs: Additional keyword arguments for the MLP.
        """
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.optimizer_type = optimizer_type if optimizer_type is not None else torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.model_kwargs = kwargs        
    
    
    def training_setup(self, X, batch_metadata):
        """
        prepare the training setup for the BaseIntegration.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_metadata (array like): Batch labels of shape (n_samples,) or (n_samples, n_effects).
            epochs (int): Number of training epochs.
            **kwargs: Additional keyword arguments for training.
        """
        
        # if X is not a tensor, convert it to a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        assert X.ndim == 2, "X must be a 2D array-like structure."
        
        
        batch_metadata = pd.DataFrame(batch_metadata)
        assert len(batch_metadata) == len(X), "Batch metadata must have the same number of samples as X."
        self.n_effects = batch_metadata.shape[1]
        assert self.n_effects > 0, "Batch metadata must have at least one effect."
        self.effects = batch_metadata.columns.tolist()
        
        self.batch_dict = {}
        batch_ids = pd.DataFrame()
        for effect in self.effects:
            if batch_metadata[effect].dtype == float:
                raise NotImplementedError(f"Effect '{effect}' is of type float: still need to implement the correction method for continuous effects.")
            else:# if the column is string, object, category of integer type, convert it to integer starting from 0
                unique_keys = batch_metadata[effect].unique()
                assert len(unique_keys) > 1, f"Effect '{effect}' must have more than one unique value."
                self.batch_dict[effect] = {key: i for i, key in enumerate(unique_keys)}
                batch_ids[effect] = batch_metadata[effect].map(self.batch_dict[effect])
        batch_ids = torch.tensor(batch_ids.values, dtype=torch.int32)
        
        # create a MultiBatchesMLP model for each effect
        self.models = {
            effect: MultiBatchesMLP(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                n_batches=len(self.batch_dict[effect]), # number of unique ids for this effect
                **self.model_kwargs
            )
            for effect in self.effects
        }
        
        # initialize the optimizer in order to manage parameters from all models
        self.optimizer = self.optimizer_type(
            [param for model in self.models.values() for param in model.parameters()],
            **self.optimizer_kwargs
        )

        # reset the per-effect similarity/adjacency cache: run_epoch fills this in lazily
        # on its first call and reuses it across epochs since X and batch_ids are constant
        # for full-batch training. A fresh fit() call means fresh inputs, so drop any
        # cache left over from a previous run.
        self._epoch_cache = None

        return batch_ids
    
    
    def fit(self, X, batch_metadata, epochs: int, k_inter: int = 5, k_intra: int = None, loss_weights: dict = None, norm: int = 2, **kwargs):
        """
        Fit the BaseIntegration model.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_metadata (array like): Batch labels of shape (n_samples,) or (n_samples, n_effects).
            epochs (int): Number of training epochs.
            k_inter (int, optional): Number of inter-batch neighbors to use (for knn loss).
            k_intra (int, optional): Number of intra-batch neighbors to use (for topology loss).
            loss_weights (dict, optional): Weights for different loss components.
            norm (int): The norm to use for the loss calculation.
        """
        
        batch_ids = self.training_setup(X, batch_metadata)
        
        if loss_weights is None:
            loss_weights = {
                "abs_lisa_loss": 1.0,
                "topology_loss": 1.0,
                "knn_loss": 1.0
            }
        
        for epoch in range(epochs):
            self.run_epoch(X, batch_ids, loss_weights=loss_weights, norm=norm, k_inter=k_inter, k_intra=k_intra, **kwargs)
        
        return
    
    def forward(self, X, batch_ids):
        """
        Forward pass through the BaseIntegration model.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).

        Returns:
            dict: A dictionary containing the transformed outputs for each effect.
        """
        
        outputs = {}
        for effect in self.effects:
            outputs[effect] = self.models[effect](X, batch_ids[:, self.effects.index(effect)])
        
        return outputs
    
    
    
    
    
    def loss(self, X, batch_ids, norm: int = 2):
        """
        Compute the multi-effect loss.

        MultiBatchesMLP.loss is built for a single effect and can't be reused here: each
        effect's model acts independently on X, and we treat the combined correction as
        the average of their outputs (rather than, say, each effect's own similarity
        matrix), so LISA/topology/knn scores are all computed once on that shared
        transformed representation, against each effect's own batch assignment and
        adjacency graphs.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).
            norm (int): The norm to use for the loss calculation.

        Returns:
            dict: Loss components, averaged across effects.
        """

        outputs = self.forward(X, batch_ids)
        x_transformed = torch.stack(list(outputs.values()), dim=0).mean(dim=0)
        w_transformed = similarity_matrix(x_transformed, norm=norm)

        abs_lisa_losses = []
        topology_losses = []
        knn_losses = []
        for effect in self.effects:
            batch_col = batch_ids[:, self.effects.index(effect)]

            abs_lisa_losses.append(torch.abs(LIBSA(w_transformed, batch_col, norm=norm)).mean())

            topology_scores = batch_similarity_preservation(
                self._epoch_cache["w"], w_transformed, batch_col, self._epoch_cache["a_intra"][effect]
            )
            topology_losses.append((-topology_scores).mean())

            knn_similarity = w_transformed[self._epoch_cache["a_inter"][effect] > 0]
            knn_losses.append((-knn_similarity).mean())

        return {
            "abs_lisa_loss": torch.stack(abs_lisa_losses).mean(),
            "topology_loss": torch.stack(topology_losses).mean(),
            "knn_loss": torch.stack(knn_losses).mean(),
        }

    def run_epoch(self, X, batch_ids, loss_weights: dict, norm: int = 2, k_inter: int = 5, k_intra: int = None, **kwargs):
        """
        Run a single training epoch.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).
            loss_weights (dict): Weights for different loss components.
            norm (int): The norm to use for the loss calculation.
            k_inter (int, optional): Number of inter-batch neighbors to use (for knn loss).
            k_intra (int, optional): Number of intra-batch neighbors to use (for topology loss).
            **kwargs: Additional keyword arguments for training.
        """

        # X and batch_ids are constant across epochs for full-batch training, so the raw
        # similarity matrix and the per-effect adjacency graphs only need computing once,
        # on the first call, and are then reused for every subsequent epoch. They're
        # written straight into self._epoch_cache (never aliased to a local variable) since
        # these are large N x N tensors and extra references invite extra copies. Subclasses
        # that train on minibatches (where these inputs change every epoch) should override
        # run_epoch instead of relying on this cache.
        if self._epoch_cache is None:
            self._epoch_cache = {"w": similarity_matrix(X, norm=norm), "a_inter": {}, "a_intra": {}}
            for effect in self.effects:
                batch_col = batch_ids[:, self.effects.index(effect)]
                self._epoch_cache["a_inter"][effect] = inter_batch_graph(X, batch_col, k=k_inter)
                self._epoch_cache["a_intra"][effect] = intra_batch_graph(X, batch_col, k=k_intra) if k_intra is not None else None

        self.optimizer.zero_grad()

        loss_dict = self.loss(X, batch_ids, norm=norm)
        total_loss = sum(loss_dict[loss_name] * weight for loss_name, weight in loss_weights.items())

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
