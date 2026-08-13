import torch
from torchvision.ops import MLP

from bibola.spatial_autocorrelation import LIBSA, similarity_matrix
from bibola.topology import batch_similarity_preservation
from bibola.knn import inter_batch_graph, intra_batch_graph

class BatchMLP(torch.nn.Module):
    
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, **kwargs):
        """
        Initialize the BatchMLP.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (list): List of hidden channel sizes.
            out_channels (int): Number of output channels.
        """
        super(BatchMLP, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels, affine=False),
            MLP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                **kwargs
            ),
            torch.nn.Linear(hidden_channels[-1], out_channels),
        )
    
    def forward(self, x):
        return self.sequential(x)
        


class DecoderMLP(torch.nn.Module):

    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, batch_ohe_dim: int = 0, **kwargs):
        """
        Initialize the DecoderMLP, mapping a batch-corrected representation back to the
        original feature space so a reconstruction loss can be computed against it.

        Args:
            in_channels (int): Number of input channels (the integration model's out_channels).
            hidden_channels (list): List of hidden channel sizes.
            out_channels (int): Number of output channels (the original input's in_channels).
            batch_ohe_dim (int): Width of the concatenated batch one-hot vector (sum of
                n_batches across all effects) that is appended to the input at the first
                layer, so the decoder knows which batch a sample came from when reconstructing it.
        """
        super(DecoderMLP, self).__init__()
        self.batch_ohe_dim = batch_ohe_dim
        self.sequential = torch.nn.Sequential(
            MLP(
                in_channels=in_channels + batch_ohe_dim,
                hidden_channels=hidden_channels,
                **kwargs
            ),
            torch.nn.Linear(hidden_channels[-1], out_channels),
        )

    def forward(self, x, batch_ohe):
        return self.sequential(torch.cat([x, batch_ohe], dim=1))


class MultiBatchesMLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, n_batches: int, **kwargs):
        """
        Initialize the MultiBatchesMLP.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (list): List of hidden channel sizes.
            out_channels (int): Number of output channels.
            n_batches (int): Number of batches.
            **kwargs: Additional keyword arguments for the MLP.
        """
        super(MultiBatchesMLP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_batches = n_batches

        self.mlps = torch.nn.ModuleList([
            BatchMLP(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                **kwargs
            )
            for _ in range(n_batches)
        ])
        
        self.initialize_weights()  # Initialize weights for all MLPs
    
    
    def initialize_weights(self):     
        # Apply Kaiming Normal initialization to the FIRST MLP
        for module in self.mlps[0].modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        # Clone the exact state of the first MLP into all other MLPs
        base_state_dict = self.mlps[0].state_dict()
        for i in range(1, self.n_batches):
            self.mlps[i].load_state_dict(base_state_dict)
        

    def forward(self, x, batch):
        outputs = torch.zeros(x.size(0), self.out_channels, device=x.device)
        batch_outputs = []
        batch_indices = []
        for b in range(self.n_batches):
            mask = (batch == b)
            if mask.any():
                batch_outputs.append(self.mlps[b](x[mask]))
                batch_indices.append(mask.nonzero(as_tuple=True)[0])
        if batch_outputs:
            all_outputs = torch.cat(batch_outputs, dim=0)
            all_indices = torch.cat(batch_indices, dim=0)
            outputs = outputs.scatter(0, all_indices.unsqueeze(1).expand_as(all_outputs), all_outputs)
        return outputs
        



