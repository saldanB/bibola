# BIBOLA: Batch Integration Based on Local Autocorrelation

**BIBOLA** is a deep learning-based batch integration framework designed to correct batch effects across multiple datasets (e.g., in single-cell genomics, spatial transcriptomics, or metabolomics). It transforms batch-affected feature spaces into a unified, batch-corrected representation.

The framework trains a multi-layer perceptron (MLP) model utilizing three distinct loss functions to optimize the integration process:
1. **Batch Entropy / Spatial Association Minimization (LIBSA)**: Minimizes local batch autocorrelation to ensure batches are uniformly mixed in the integrated space.
2. **Intra-batch Geometry Preservation (Topology Loss)**: Preserves the original subpopulation topologies and local distances within each individual batch.
3. **Inter-batch k-NN Attraction Loss**: Attracts biologically similar samples that are close in the initial space but originate from different batches.

---

## Methodology & Loss Formulation

BIBOLA structures batch correction as a joint optimization problem over three objectives:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{LIBSA}} + \lambda_2 \mathcal{L}_{\text{Topology}} + \lambda_3 \mathcal{L}_{\text{Attraction}}$$

### 1. LIBSA Loss (Minimizing Batch Spatial Autocorrelation)
Local Indicators of Batch Spatial Association (LIBSA) quantify spatial clustering patterns of batch membership.
- For each cell/sample, LIBSA computes spatial autocorrelation of the batch identity indicator.
- Minimizing the absolute LIBSA values ($\mathcal{L}_{\text{LIBSA}} = \frac{1}{N}\sum |I_i|$) forces batch membership to become spatially random in the integrated representation, eliminating batch separating structures.

### 2. Intra-batch Topology Loss
To ensure that batch correction does not distort the intrinsic biological structure/signal of individual datasets:
- The Pearson correlation between the original and transformed similarity matrices (weighted by the intra-batch k-NN graph if required) is computed for each batch.
- Minimizing negative correlation ($\mathcal{L}_{\text{Topology}} = -\text{Mean}(\text{Pearson Correlation})$) preserves intra-batch local and manifold configurations.

### 3. Inter-batch k-NN Attraction Loss
To align equivalent subpopulations across different batches:
- A k-NN graph is constructed over distinct batch boundaries in the original space ($A_{\text{inter}}$).
- The transformed similarity is maximized for these cross-batch edges ($\mathcal{L}_{\text{Attraction}} = -\text{Mean}(W_{\text{transformed}}[A_{\text{inter}} > 0])$), bringing matched subpopulations into alignment.

## Quickstart

Verify the installation and correct your data using the workflow below (taken from `workflow_with_simulated_data.ipynb`):

```python
import torch
from bibola.MLP import MultiBatchesMLP
from bibola.spatial_autocorrelation import similarity_matrix

# 1. Prepare inputs (X: [N, D] tensor of features, batch: [N] tensor of 0-indexed integers)
# X = ...
# batch = ... 

# 2. Instantiate the Multi-Batch MLP model
model = MultiBatchesMLP(
    in_channels=X.size(1),
    hidden_channels=[32],     # MLP hidden sizes
    out_channels=2,            # Integrated target dimension
    n_batches=batch.max().item() + 1,
    dropout=0.1
)

# 3. Fit the model jointly using the custom combined loss 
model.fit(
    X, 
    batch, 
    epochs=25, 
    lr=1e-4, 
    k_inter=5,       # Inter-batch neighbours to align 
    k_intra=50,       # Intra-batch neighbours to preserve 
    norm=1,
    loss_weights={
        "abs_lisa_loss": 1.0, 
        "topology_loss": 1.0, 
        "knn_loss": 1.0
    }
)

# 4. Integrate / Correct the data
model.eval()
with torch.no_grad():
    X_transformed = model(X, batch)
```

For a full interactive demonstration with synthetic datasets and t-SNE projections, please refer to the notebook [workflow_with_simulated_data.ipynb].