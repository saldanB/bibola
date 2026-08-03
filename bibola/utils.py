import torch

def extract_flattened_triangle(
    matrix: torch.Tensor, 
    upper: bool = True, 
    include_diagonal: bool = False
) -> torch.Tensor:
    """
    Extracts the flattened upper or lower triangular values from a 2D matrix 
    (or batch of 2D matrices).

    Args:
        matrix (torch.Tensor): Square matrix of shape (N, N) or batch of matrices (B, N, N).
        upper (bool): If True, extracts upper triangle. If False, extracts lower triangle. Default: True.
        include_diagonal (bool): If True, includes the main diagonal. Default: False.

    Returns:
        torch.Tensor: Flattened 1D tensor of shape (N*(N-1)/2,) or (N*(N+1)/2,).
                      For batch inputs (B, N, N), returns shape (B, K).
    """
    if matrix.dim() not in (2, 3):
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(matrix.shape)}")
    
    N = matrix.shape[-1]
    if matrix.shape[-2] != N:
        raise ValueError(f"Expected a square matrix, got shape {tuple(matrix.shape)}")

    # 1. Determine diagonal offset
    # offset > 0 shifts right/up (excludes diagonal for upper)
    # offset < 0 shifts left/down (excludes diagonal for lower)
    if upper:
        offset = 0 if include_diagonal else 1
        row_idx, col_idx = torch.triu_indices(N, N, offset=offset, device=matrix.device)
    else:
        offset = 0 if include_diagonal else -1
        row_idx, col_idx = torch.tril_indices(N, N, offset=offset, device=matrix.device)

    # 2. Extract values based on tensor dimensions
    if matrix.dim() == 2:
        return matrix[row_idx, col_idx]
    else:
        # Handles batch dimension (B, N, N) -> (B, K)
        return matrix[:, row_idx, col_idx]