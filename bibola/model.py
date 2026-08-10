import torch
import pandas as pd
from bibola.MLP import MultiBatchesMLP
from bibola.spatial_autocorrelation import similarity_matrix


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
        
        return batch_ids
    
    
    def fit(self, X, batch_metadata, epochs: int, loss_weights: dict = None, **kwargs):
        """
        Fit the BaseIntegration model.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_metadata (array like): Batch labels of shape (n_samples,) or (n_samples, n_effects).
            epochs (int): Number of training epochs.
            loss_weights (dict, optional): Weights for different loss components.
        """
        
        batch_ids = self.training_setup(X, batch_metadata)
        
        if loss_weights is None:
            loss_weights = {
                "abs_lisa_loss": 1.0,
                "topology_loss": 1.0,
                "knn_loss": 1.0
            }
        
        for epoch in range(epochs):
            self.run_epoch(X, batch_ids, loss_weights=loss_weights, **kwargs)
        
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
    
    
    
    
    def run_epoch(self, X, batch_ids, loss_weights: dict, **kwargs):
        """
        Run a single training epoch.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).
            loss_weights (dict): Weights for different loss components.
            **kwargs: Additional keyword arguments for training.
        """
        
        self.optimizer.zero_grad()
        