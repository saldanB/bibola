import warnings
import torch

def distance_matrix(X: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """
    Compute the pairwise Euclidean distance matrix for a set of points.

    Args:
        X (torch.Tensor): A tensor of shape (n_samples, n_features) representing the input points.
        norm (int): The order of the norm (default is 2 for Euclidean distance).

    Returns:
        torch.Tensor: A tensor of shape (n_samples, n_samples) representing the pairwise Euclidean distance matrix.
    """
    # Compute the squared differences
    diff = X.unsqueeze(1) - X.unsqueeze(0)
    # Compute the squared distances given the specified norm
    distances = torch.norm(diff, p=norm, dim=-1)
    
    return distances


def similarity_matrix(X: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """
    Compute the pairwise similarity matrix for a set of points based on the distance matrix.

    Args:
        X (torch.Tensor): A tensor of shape (n_samples, n_features) representing the input points.
        norm (int): The order of the norm (default is 2 for Euclidean distance).

    Returns:
        torch.Tensor: A tensor of shape (n_samples, n_samples) representing the pairwise similarity matrix.
    """
    # Compute the distance matrix
    distances = distance_matrix(X, norm=norm)
    # Convert distances to similarities (e.g., using a Gaussian kernel)
    sigma = distances.std()
    similarities = torch.exp(-distances**2 / (2 * sigma**2))
    
    return similarities


def LIBSA(W: torch.Tensor, batch: torch.Tensor, norm: int = 2) -> torch.Tensor:
    """
    Compute the Local Indicators of Batch Spatial Association (LIBSA) for each sample in a batch.

    Args:
        W (torch.Tensor): A tensor of shape (n_samples, n_samples) representing the spatial weights matrix.
        batch (torch.Tensor): A tensor of shape (n_samples,) representing the batch indices.
        norm (int): The order of the norm (default is 2 for Euclidean distance).

    Returns:
        torch.Tensor: A tensor of shape (n_samples, n_batches) representing the computed LISA values for each sample and batch.
    """
    
    assert batch.dim() == 1, "batch must be a 1D tensor"
    assert W.size(0) == batch.size(0), "W and batch must have the same number of samples"
    assert batch.dtype == torch.int32, "batch must be a tensor of integers"
    assert (batch.min() == 0) and (batch.max() == batch.unique().size(0) - 1), "batch must be progressively numbered from 0 to n_batches - 1"
    
    n_batches = batch.max() + 1
    
    # Zero out diagonal without in-place mutation (preserves autograd graph)
    diag_mask = ~torch.eye(W.size(0), dtype=torch.bool, device=W.device)
    W = W * diag_mask
    W = W / W.sum(dim=1, keepdim=True)  # Row-normalize the spatial weights matrix
    
    columns = []
    for b in range(n_batches):
        y = (batch == b).float()
        y_mean = y.mean()
        y_std = y.std(unbiased=False)
        # Handle edge case where y is constant (std == 0)
        if y_std == 0:
            warnings.warn(f"Batch {b} has constant values, resulting in zero standard deviation. LISA values will be zero for this batch.")
            columns.append(torch.zeros(W.size(0), dtype=torch.float32, device=W.device))
            continue
        z = (y - y_mean) / y_std
        spatial_lag = torch.matmul(W, z)
        local_I = z * spatial_lag
        columns.append(local_I)
    
    L = torch.stack(columns, dim=1)
    return L