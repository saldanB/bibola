import torch
import pandas as pd
from bibola.MLP import MultiBatchesMLP, DecoderMLP
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


    def __repr__(self):
        status = f"fitted, effects={self.effects}" if hasattr(self, "effects") else "not fitted"
        return (f"{self.__class__.__name__}(in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, out_channels={self.out_channels}, "
                f"optimizer={self.optimizer_type.__name__}, {status})")


    def __str__(self):
        lines = [f"{self.__class__.__name__}"]
        lines.append(f"  in_channels:     {self.in_channels}")
        lines.append(f"  hidden_channels: {self.hidden_channels}")
        lines.append(f"  out_channels:    {self.out_channels}")
        lines.append(f"  optimizer:       {self.optimizer_type.__name__}({self.optimizer_kwargs})")
        if self.model_kwargs:
            lines.append(f"  model_kwargs:    {self.model_kwargs}")

        if hasattr(self, "effects"):
            n_params = sum(p.numel() for model in list(self.models.values()) + [self.decoder] for p in model.parameters())
            lines.append(f"  status:          fitted ({n_params:,} parameters)")
            lines.append("  effects:")
            for effect in self.effects:
                lines.append(f"    - {effect}: {len(self.batch_dict[effect])} batches")
        else:
            lines.append("  status:          not fitted (call fit() or training_setup())")

        return "\n".join(lines)


    def set_mode(self, mode: str):
        """
        Set the mode of the model.

        Args:
            mode (str): Mode to set ('train' or 'eval').
        """
        for model in list(self.models.values()) + [self.decoder]:
            if mode == 'train':
                model.train()
            elif mode == 'eval':
                model.eval()
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'train' or 'eval'.")
    
    
    def training_setup(self, X, batch_metadata, reinitialize: bool = False):
        """
        prepare the training setup for the BaseIntegration.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_metadata (array like): Batch labels of shape (n_samples,) or (n_samples, n_effects).
            epochs (int): Number of training epochs.
            reinitialize (bool): Whether to rebuild (fresh-initialize) the per-effect models and
                the optimizer. Skipped when models already exist and reinitialize is False, so a
                second fit() call on the same instance continues training the existing weights.
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
        self.batch_ohe_dim = sum(len(self.batch_dict[effect]) for effect in self.effects)

        if reinitialize or not hasattr(self, "models"):
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

            # decoder maps the corrected representation (out_channels) back to the
            # original feature space (in_channels), for the reconstruction loss term.
            # It's also given the sample's batch one-hot (concatenated across effects),
            # so reconstruction is conditioned on which batch the sample came from.
            self.decoder = DecoderMLP(
                in_channels=self.out_channels,
                hidden_channels=self.hidden_channels[::-1],
                out_channels=self.in_channels,
                batch_ohe_dim=self.batch_ohe_dim,
                **self.model_kwargs
            )

            # initialize the optimizer in order to manage parameters from all models
            self.optimizer = self.optimizer_type(
                [param for model in list(self.models.values()) + [self.decoder] for param in model.parameters()],
                **self.optimizer_kwargs
            )
            
            self._loss_cache = []  # reset the loss cache when reinitializing models

        # reset the per-effect similarity/adjacency cache: run_epoch fills this in lazily
        # on its first call and reuses it across epochs since X and batch_ids are constant
        # for full-batch training. A fresh fit() call means fresh inputs, so drop any
        # cache left over from a previous run.
        self._epoch_cache = None

        return X, batch_ids
    
    
    @staticmethod
    def _format_loss(loss_dict) -> str:
        return ", ".join(f"{loss_name}={loss_value:.4f}" for loss_name, loss_value in loss_dict.items())

    def fit(self, X, batch_metadata, epochs: int, k_inter: int = 5, k_intra: int = None,
            loss_weights: dict = None, norm: int = 2, reinitialize: bool = False,
            print_every: int = None, **kwargs):
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
            reinitialize (bool): Whether to reinitialize the model parameters before training.
            print_every (int, optional): Print the epoch loss every print_every epochs.
                None (default) disables printing.
        """

        X, batch_ids = self.training_setup(X, batch_metadata, reinitialize=reinitialize)

        if loss_weights is None:
            loss_weights = {
                "abs_lisa_loss": 1.0,
                "topology_loss": 1.0,
                "knn_loss": 1.0,
                "reconstruction_loss": 1.0
            }

        for epoch in range(epochs):
            total_loss = self.run_epoch(
                X, batch_ids, loss_weights=loss_weights, norm=norm,
                k_inter=k_inter, k_intra=k_intra, **kwargs
            )
            if print_every is not None and (epoch % print_every == 0 or epoch == epochs - 1):
                components = self._format_loss(self._loss_cache[-1]["loss"])
                print(f"Epoch {epoch + 1}/{epochs} - total_loss={total_loss:.4f} ({components})")

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


    def transform(self, X, batch_metadata):
        """
        Apply the trained model to produce the batch-corrected representation.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_metadata (array like): Batch labels of shape (n_samples,) or (n_samples, n_effects),
                using the same effect columns and label values seen during training.

        Returns:
            torch.Tensor: Corrected representation of shape (n_samples, out_channels), the
                average across per-effect model outputs (same combination rule as loss()).
        """

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        assert X.ndim == 2, "X must be a 2D array-like structure."

        batch_metadata = pd.DataFrame(batch_metadata)
        assert list(batch_metadata.columns) == self.effects, "batch_metadata must contain the same effects used during training."

        batch_ids = pd.DataFrame({
            effect: batch_metadata[effect].map(self.batch_dict[effect]) for effect in self.effects
        })
        assert not batch_ids.isna().any().any(), "batch_metadata contains label(s) not seen during training."
        batch_ids = torch.tensor(batch_ids.values, dtype=torch.int32)

        self.set_mode('eval')
        with torch.no_grad():
            outputs = self.forward(X, batch_ids)

        return torch.stack(list(outputs.values()), dim=0).mean(dim=0)


    def _batch_ohe(self, batch_ids):
        """
        Build the concatenated batch one-hot encoding for the decoder.

        Each effect's batch id column is one-hot encoded independently (width =
        that effect's number of unique batches) and the results are concatenated
        along the feature dimension, in `self.effects` order. E.g. for a sample in
        run 3 (of 3) and donor 2 (of 6): ohe = [0,0,1] ++ [0,1,0,0,0,0].

        Args:
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).

        Returns:
            torch.Tensor: One-hot batch encoding of shape (n_samples, batch_ohe_dim).
        """
        ohe_parts = [
            torch.nn.functional.one_hot(
                batch_ids[:, self.effects.index(effect)].long(),
                num_classes=len(self.batch_dict[effect])
            ).float()
            for effect in self.effects
        ]
        return torch.cat(ohe_parts, dim=1)

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

        batch_ohe = self._batch_ohe(batch_ids)
        x_reconstructed = self.decoder(x_transformed, batch_ohe)
        # relative (scale-free) reconstruction error, so its magnitude stays comparable
        # to the other, O(1) similarity-based loss terms regardless of X's raw scale
        reconstruction_loss = (torch.norm(X - x_reconstructed, p=norm, dim=1)
                                / (torch.norm(X, p=norm, dim=1) + 1e-8)).mean()

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
            "reconstruction_loss": reconstruction_loss,
        }

    def run_epoch(self, X, batch_ids, loss_weights: dict, norm: int = 2,
                  k_inter: int = 5, k_intra: int = None, verbose: bool = False, **kwargs):
        """
        Run a single training epoch.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).
            loss_weights (dict): Weights for different loss components.
            norm (int): The norm to use for the loss calculation.
            k_inter (int, optional): Number of inter-batch neighbors to use (for knn loss).
            k_intra (int, optional): Number of intra-batch neighbors to use (for topology loss).
            verbose (bool): Whether to print the loss breakdown for this epoch.
            **kwargs: Additional keyword arguments for training.
        """
        
        self.set_mode('train')

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
        total_loss /= sum(loss_weights.values()) # normalize by the sum of weights so that the absolute scale of the weights doesn't matter
        
        self._loss_cache.append({})  # store the loss components for this epoch
        self._loss_cache[-1]["loss"] = {loss_name: loss_value.item()for loss_name, loss_value in loss_dict.items()}
        self._loss_cache[-1]["loss_weights"] = loss_weights

        if verbose:
            print(f"  loss: {self._format_loss(self._loss_cache[-1]['loss'])}")

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
    
    @property
    def loss_history(self):
        return pd.DataFrame.from_dict({
            (outer, inner): [row[outer][inner] for row in self._loss_cache]
            for outer in self._loss_cache[0]
            for inner in self._loss_cache[0][outer]
        }, orient="columns")
    
    
class ChunkIntegration(BaseIntegration):
    """
    A subclass of BaseIntegration that performs chunked training.
    Particularly useful for large datasets whose adjacencys and similarity matrices do not fit in memory.
    The n_chunks parameter controls the number of chunks used for training.
    """
    
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int,
                 optimizer_type: torch.optim.Optimizer = None, optimizer_kwargs: dict = None,
                 **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels,
                         optimizer_type=optimizer_type, optimizer_kwargs=optimizer_kwargs,
                         **kwargs)
        
    def _stratified_chunks(self, batch_ids, n_chunks: int, random_seed: int = None):
        """
        Partition sample indices into n_chunks groups, each homogeneous in batch_ids
        composition for every effect (stratified split).

        Samples are grouped by their joint (per-effect) batch label combination, each
        group is shuffled, and its members are dealt round-robin across the n_chunks
        buckets, so every chunk gets close to the same proportion of every batch id,
        for every effect, as the full dataset.

        Args:
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).
            n_chunks (int): Number of chunks to produce.
            random_seed (int, optional): Seed for the shuffle within each stratum.

        Returns:
            list[torch.Tensor]: n_chunks tensors of sample indices.
        """
        generator = torch.Generator()
        if random_seed is not None:
            # offset by a call counter so repeated epochs (same seed passed every
            # time by fit()) don't re-draw the exact same chunks, while staying
            # reproducible run-to-run.
            self._chunk_call_count = getattr(self, "_chunk_call_count", 0) + 1
            generator.manual_seed(random_seed + self._chunk_call_count)
        else:
            generator.seed()

        composite_key = torch.zeros(batch_ids.size(0), dtype=torch.int64)
        for effect in self.effects:
            n_batches_effect = len(self.batch_dict[effect])
            composite_key = composite_key * n_batches_effect + batch_ids[:, self.effects.index(effect)].to(torch.int64)

        chunks = [[] for _ in range(n_chunks)]
        for key in composite_key.unique():
            group_indices = (composite_key == key).nonzero(as_tuple=True)[0]
            shuffled = group_indices[torch.randperm(group_indices.size(0), generator=generator)]
            for i, idx in enumerate(shuffled.tolist()):
                chunks[i % n_chunks].append(idx)

        return [torch.tensor(chunk, dtype=torch.long) for chunk in chunks]

    def run_epoch(self, X, batch_ids, loss_weights: dict, norm: int = 2,
                  k_inter: int = 5, k_intra: int = None, n_chunks: int = 10,
                  random_seed : int = None, verbose: bool = False, **kwargs):
        """
        Run a single training epoch with chunked processing.

        Chunks are re-drawn (stratified by batch_ids) on every call, since a fresh
        partition is what makes chunked training approximate full-batch training over
        many epochs. Each chunk gets its own similarity matrix and adjacency graphs
        (too expensive to keep N x N tensors for every chunk of every epoch around),
        and its own backward()/step() call, so one epoch performs n_chunks optimizer
        updates instead of one.

        Args:
            X (array like): Input data of shape (n_samples, n_features).
            batch_ids (torch.Tensor): Batch IDs of shape (n_samples, n_effects).
            loss_weights (dict): Weights for different loss components.
            norm (int): The norm to use for the loss calculation.
            k_inter (int, optional): Number of inter-batch neighbors to use (for knn loss).
            k_intra (int, optional): Number of intra-batch neighbors to use (for topology loss).
            n_chunks (int): Number of chunks to split the data into for processing.
            random_seed (int, optional): Random seed for reproducibility.
            verbose (bool): Whether to print the loss breakdown for each chunk.
            **kwargs: Additional keyword arguments for training.
        """

        self.set_mode('train')

        chunk_indices = self._stratified_chunks(batch_ids, n_chunks, random_seed=random_seed)

        chunk_losses = {}
        chunk_total_losses = []
        for i, indices in enumerate(chunk_indices):
            if indices.numel() == 0:
                continue

            X_chunk = X[indices]
            batch_ids_chunk = batch_ids[indices]

            # per-chunk similarity/adjacency: chunk membership changes every epoch, so
            # nothing here is worth caching across calls the way BaseIntegration does.
            a_inter_chunk, a_intra_chunk = {}, {}
            for effect in self.effects:
                batch_col = batch_ids_chunk[:, self.effects.index(effect)]
                a_inter_chunk[effect] = inter_batch_graph(X_chunk, batch_col, k=k_inter)
                a_intra_chunk[effect] = intra_batch_graph(X_chunk, batch_col, k=k_intra) if k_intra is not None else None
            self._epoch_cache = {"w": similarity_matrix(X_chunk, norm=norm), "a_inter": a_inter_chunk, "a_intra": a_intra_chunk}

            self.optimizer.zero_grad()
            loss_dict = self.loss(X_chunk, batch_ids_chunk, norm=norm)
            total_loss = sum(loss_dict[loss_name] * weight for loss_name, weight in loss_weights.items())
            total_loss /= sum(loss_weights.values())
            total_loss.backward()
            self.optimizer.step()

            for loss_name, loss_value in loss_dict.items():
                chunk_losses.setdefault(loss_name, []).append(loss_value.item())
            chunk_total_losses.append(total_loss.item())

            if verbose:
                components = self._format_loss({name: value.item() for name, value in loss_dict.items()})
                print(f"  chunk {i + 1}/{n_chunks} - total_loss={total_loss.item():.4f} ({components})")

        self._epoch_cache = None  # nothing carries over: next epoch draws a fresh partition

        self._loss_cache.append({})
        self._loss_cache[-1]["loss"] = {loss_name: sum(values) / len(values) for loss_name, values in chunk_losses.items()}
        self._loss_cache[-1]["loss_weights"] = loss_weights

        return sum(chunk_total_losses) / len(chunk_total_losses)
