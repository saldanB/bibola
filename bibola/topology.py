import torch

from bibola.utils import extract_flattened_triangle

def similarity_correlation(W: torch.Tensor, W_transformed: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the correlation between the original and transformed similarity matrices.
    
    Args:
        W (torch.Tensor): The original similarity matrix.
        W_transformed (torch.Tensor): The transformed similarity matrix.
        adj (torch.Tensor): The adjacency matrix inside the batch.
    """
    
    if adj is None: # take into account all distances among the batch
        # Flatten the distance matrices
        original_flat = extract_flattened_triangle(W, upper=False, include_diagonal=False)
        transformed_flat = extract_flattened_triangle(W_transformed, upper=False, include_diagonal=False)
    else:
        # Use the adjacency matrix to mask the distance matrices
        original_flat = W[adj > 0]
        transformed_flat = W_transformed[adj > 0]

    # Compute the correlation coefficient
    correlation_matrix = torch.corrcoef(torch.stack((original_flat, transformed_flat)))
    correlation_coefficient = correlation_matrix[0, 1]

    return correlation_coefficient


def batch_similarity_preservation(W: torch.Tensor, W_transformed: torch.Tensor, batch: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the average similarity preservation for each batch.

    Args:
        W (torch.Tensor): The original similarity matrix.
        W_transformed (torch.Tensor): The transformed similarity matrix.
        batch (torch.Tensor): A tensor indicating the batch assignment for each sample.
        adj (torch.Tensor): The adjacency matrix inside the batch.

    Returns:
        torch.Tensor: A tensor containing the similarity preservation score for each batch.
    """
    
    
    n_batches = torch.unique(batch)
    
    assert batch.dim() == 1, "batch must be a 1D tensor"
    assert W.size(0) == batch.size(0), "W and batch must have the same number of samples"
    assert batch.dtype == torch.int32, "batch must be a tensor of integers"
    assert (batch.min() == 0) and (batch.max() == batch.unique().size(0) - 1), "batch must be progressively numbered from 0 to n_batches - 1"
    
    scores = []

    for b in n_batches:
        if adj is not None:
            batch_adj = adj[batch == b][:, batch == b]
        # Get indices of the current batch
        indices = (batch == b).nonzero(as_tuple=True)[0]
        
        # Extract the submatrices for the current batch
        W_batch = W[indices][:, indices]
        W_transformed_batch = W_transformed[indices][:, indices]
        
        # Compute the correlation for the current batch
        score = similarity_correlation(W_batch, W_transformed_batch, batch_adj)
        scores.append(score)

    preservation_scores = torch.stack(scores)
    return preservation_scores
