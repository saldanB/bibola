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
            **kwargs: Additional keyword arguments for the MLP.
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
        
    
    def loss(self, x, w, a_inter, batch, a_intra=None, norm=2):
        
        x_transformed = self.forward(x, batch)
        w_transformed = similarity_matrix(x_transformed, norm=norm)
        
        # Compute the Local Indicators of Batch Spatial Association (LIBSA) for each sample in a batch
        # we want to minimize the absolute values of the LISA values, as we don't want negative autocorrelation.
        abs_lisa_values = torch.abs(LIBSA(w_transformed, batch, norm=norm)) # (n_samples, n_batches))
        
        # Compute the batch similarity preservation scores
        # it quantifies how well the similarity structure of each batch is preserved after transformation.
        topology_preservation_scores = batch_similarity_preservation(w, w_transformed, batch, a_intra) # (n_batches,)
        
        knn_similarity = w_transformed[a_inter>0]
        
        return {
            "abs_lisa_loss": abs_lisa_values.mean(),
            "topology_loss": (-topology_preservation_scores).mean(),
            "knn_loss": (-knn_similarity).mean()
        }

    def fit(self, x, batch, epochs, norm=2, lr=1e-3, print_every=10, k_inter=5, k_intra=None):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # original intra-batch topology we wanna preserve after transformation
        w = similarity_matrix(x, norm=norm)
        
        # adjacency matrix of the inter-batch graph, where edges connect samples from different batches based on their similarity.
        a_inter = inter_batch_graph(x, batch, k=k_inter)
        
        # adjacency matrix of the intra-batch graph, where edges connect samples from the same batch based on their similarity.
        a_intra = intra_batch_graph(x, batch, k=k_intra) if k_intra is not None else None
        
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(epochs):
                optimizer.zero_grad()
                loss_dict = self.loss(x, w, a_inter, batch, a_intra=a_intra, norm=norm)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()
                if epoch % print_every == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
                  
        
        
        
        
        