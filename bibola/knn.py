
import torch

from bibola.spatial_autocorrelation import distance_matrix

import torch

def inter_batch_graph(X: torch.Tensor, batch: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Compute the inter-batch graph by iterating through all permutations of distinct batch pairs (b1, b2),
    finding the k-nearest neighbors from each node in b1 to nodes in b2.

    Args:
        X (torch.Tensor): Input data of shape (N, D).
        batch (torch.Tensor): Batch indices of shape (N,).
        k (int): Number of nearest neighbors per target batch.

    Returns:
        torch.Tensor: Binary adjacency matrix of shape (N, N).
    """
    N = X.shape[0]
    unique_batches = torch.unique(batch)
    
    # 1. Compute global distance matrix
    # we don't mind the norm since the order of distances is preserved across norms.
    dist_matrix = distance_matrix(X, norm=1)  # shape (N, N)

    # 2. Collect edges out-of-place using lists to maintain clean autograd history
    all_rows = []
    all_cols = []

    # 3. Permute over all batch pairs (b1, b2)
    for b1 in unique_batches:
        idx1 = (batch == b1).nonzero(as_tuple=True)[0]
        
        for b2 in unique_batches:
            if b1 == b2:
                continue  # Skip intra-batch comparisons
            
            idx2 = (batch == b2).nonzero(as_tuple=True)[0]
            
            # Slice rectangular sub-matrix of distances from b1 nodes to b2 nodes: shape (len(b1), len(b2))
            sub_dist = dist_matrix[idx1][:, idx2]
            
            # Cap k if target batch has fewer elements than k
            effective_k = min(k, idx2.size(0))
            
            # Find k-nearest neighbors in b2 for each node in b1
            # sub_knn shape: (len(b1), effective_k)
            sub_knn_idx = torch.topk(sub_dist, k=effective_k, largest=False, dim=1).indices
            
            # Map local sub-matrix indices back to global N x N indices
            global_b1_rows = idx1.unsqueeze(1).expand(-1, effective_k)
            global_b2_cols = idx2[sub_knn_idx]
            
            all_rows.append(global_b1_rows.reshape(-1))
            all_cols.append(global_b2_cols.reshape(-1))

    # 4. Construct final graph out-of-place (prevents inplace mutation errors)
    inter_graph = torch.zeros(N, N, device=X.device, dtype=X.dtype)
    
    if len(all_rows) > 0:
        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        
        # Vectorized assignment via index_put (or scatter) without modifying intermediate computational graphs
        inter_graph = inter_graph.index_put((rows, cols), torch.tensor(1.0, device=X.device))

    return inter_graph


def intra_batch_graph(X: torch.Tensor, batch: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Compute the intra-batch graph by finding the k-nearest neighbors within each batch.

    Args:
        X (torch.Tensor): Input data of shape (N, D).
        batch (torch.Tensor): Batch indices of shape (N,).
        k (int): Number of nearest neighbors per node.

    Returns:
        torch.Tensor: Binary adjacency matrix of shape (N, N).
    """
    N = X.shape[0]
    unique_batches = torch.unique(batch)
    
    # 1. Compute global distance matrix
    dist_matrix = distance_matrix(X, norm=1)  # shape (N, N)

    # 2. Collect edges out-of-place using lists to maintain clean autograd history
    all_rows = []
    all_cols = []

    # 3. Iterate over each batch
    for b in unique_batches:
        idx = (batch == b).nonzero(as_tuple=True)[0]
        
        # Slice sub-matrix of distances within the batch: shape (len(b), len(b))
        sub_dist = dist_matrix[idx][:, idx]
        
        # Cap k if batch has fewer elements than k
        effective_k = min(k, idx.size(0) - 1)  # Exclude self
        
        if effective_k <= 0:
            continue  # Skip if not enough samples in the batch
        
        # Find k-nearest neighbors within the batch (mask self so it's never picked;
        # eye * inf would give NaN off-diagonal since 0 * inf = NaN, corrupting topk)
        self_mask = torch.eye(len(idx), dtype=torch.bool, device=X.device)
        sub_knn_idx = torch.topk(sub_dist.masked_fill(self_mask, float('inf')),
                                  k=effective_k, largest=False, dim=1).indices
        
        # Map local sub-matrix indices back to global N x N indices
        global_rows = idx.unsqueeze(1).expand(-1, effective_k)
        global_cols = idx[sub_knn_idx]
        
        all_rows.append(global_rows.reshape(-1))
        all_cols.append(global_cols.reshape(-1))

    # 4. Construct final graph out-of-place
    intra_graph = torch.zeros(N, N, device=X.device, dtype=X.dtype)
    
    if len(all_rows) > 0:
        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        
        intra_graph = intra_graph.index_put((rows, cols), torch.tensor(1.0, device=X.device))
    
    return intra_graph

    
    