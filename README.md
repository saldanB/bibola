# BIBOLA: Batch Integration Based on Local Autocorrelation

**BIBOLA** is a deep learning batch integration framework designed to correct batch effects
across multiple datasets (e.g. single-cell genomics, spatial transcriptomics, or metabolomics).
It learns a transformation of a batch-affected feature space (e.g. a PCA embedding) into a
batch-corrected representation, while preserving the biological structure of the data.

Training jointly optimizes four loss terms:
1. **LIBSA loss** — minimizes local batch spatial autocorrelation, so batches end up spatially
   mixed in the corrected representation instead of forming separate clusters.
2. **Topology loss** — preserves each batch's own local similarity structure (its intra-batch
   k-NN geometry), so correction doesn't destroy real biological signal.
3. **k-NN attraction loss** — pulls similar samples from *different* batches together, using an
   inter-batch k-NN graph built in the original space.
4. **Reconstruction loss** — a decoder maps the corrected representation back to the original
   feature space; penalizing that reconstruction error keeps the correction information-
   preserving instead of collapsing everything together to satisfy the other three terms.

---

## Architecture

- **`BatchMLP`** (`bibola/MLP.py`) — a small `BatchNorm1d + MLP + Linear` block, one instance
  per batch label.
- **`MultiBatchesMLP`** (`bibola/MLP.py`) — one `BatchMLP` per unique batch label for a given
  effect (e.g. `"batch"`, `"donor"`), all initialized from the same weights, applied to each
  sample according to its own batch label.
- **`DecoderMLP`** (`bibola/MLP.py`) — maps the corrected representation back to the original
  feature space, used only to compute the reconstruction loss.
- **`BaseIntegration`** (`bibola/model.py`) — owns one `MultiBatchesMLP` per effect column in
  `batch_metadata` plus the shared `DecoderMLP`, and trains all of them jointly (full-batch)
  against the four losses above. Multiple effects (e.g. `"batch"` and `"donor"` together) are
  supported: each gets its own model, and the final corrected representation is the average of
  all effects' outputs.
- **`ChunkIntegration`** (`bibola/model.py`, subclass of `BaseIntegration`) — for datasets too
  large to hold an `N x N` similarity/adjacency matrix in memory. Each epoch re-draws a fresh
  stratified partition of the data into `n_chunks` groups (balanced per batch label) and runs
  one full backward/step per chunk, so the pairwise structures only ever need to be built at
  chunk size.

Both classes share the same `fit()` / `transform()` API; `ChunkIntegration` just overrides how
`run_epoch()` slices the data.

---

## Methodology & Loss Formulation

$$\mathcal{L}_{\text{total}} = \frac{\lambda_1 \mathcal{L}_{\text{LIBSA}} + \lambda_2 \mathcal{L}_{\text{Topology}} + \lambda_3 \mathcal{L}_{\text{Attraction}} + \lambda_4 \mathcal{L}_{\text{Reconstruction}}}{\lambda_1 + \lambda_2 + \lambda_3 + \lambda_4}$$

### 1. LIBSA loss (`abs_lisa_loss`)
Local Indicators of Batch Spatial Association quantify how spatially clustered batch membership
is around each sample, given the transformed similarity matrix. Minimizing the mean absolute
LISA value ($\mathcal{L}_{\text{LIBSA}} = \frac{1}{N}\sum |I_i|$) forces batch membership to
become spatially random in the corrected representation, i.e. removes batch-separating
structure.

### 2. Topology loss (`topology_loss`)
For each batch, the Pearson correlation between the original and transformed similarity
matrices (restricted to that batch's intra-batch k-NN graph, `k_intra`) measures how well its
internal geometry survived the correction. Minimizing the negative mean correlation
($\mathcal{L}_{\text{Topology}} = -\text{Mean}(\text{Pearson correlation})$) keeps each batch's
own manifold structure intact.

### 3. k-NN attraction loss (`knn_loss`)
An inter-batch k-NN graph (`k_inter`) connects each sample to its nearest neighbors in *other*
batches, in the original space. Maximizing the transformed similarity along those edges
($\mathcal{L}_{\text{Attraction}} = -\text{Mean}(W_{\text{transformed}}[A_{\text{inter}} > 0])$)
pulls matching subpopulations from different batches together.

### 4. Reconstruction loss (`reconstruction_loss`)
A `DecoderMLP` reconstructs the original input from the corrected representation. The loss is
the *relative* L-norm reconstruction error per sample,
$\mathcal{L}_{\text{Reconstruction}} = \text{Mean}\left(\frac{\|X - \hat{X}\|_p}{\|X\|_p + \epsilon}\right)$,
normalized by each sample's own magnitude so it stays on the same O(1) scale as the three
similarity-based losses above (a raw reconstruction error would otherwise dominate the combined
loss purely due to unit/scale differences, not because it matters more).

---

## Quickstart

BIBOLA is typically used on top of an existing `anndata`/`scanpy` object, correcting a PCA (or
Harmony) embedding and writing the result back into `.obsm` for downstream neighbors/UMAP —
this is the pattern used in `pancreas_benchmark.ipynb` and `lung_atlas_benchmark.ipynb`:

```python
import scanpy as sc
from bibola.model import ChunkIntegration

# adata.obsm["X_pca"]: [N, D] embedding to correct
# adata.obs["batch"]: categorical batch label, one column per effect to correct for

model = ChunkIntegration(
    in_channels=adata.obsm["X_pca"].shape[1],
    hidden_channels=[128],
    out_channels=50,
    dropout=0.1,
    optimizer_kwargs={"lr": 1e-3},
)

model.fit(
    adata.obsm["X_pca"], adata.obs["batch"],
    epochs=5, print_every=1,
    k_inter=5, k_intra=100, norm=1, n_chunks=5,
    loss_weights={
        "abs_lisa_loss": 1.0,
        "topology_loss": 1.0,
        "knn_loss": 1.0,
        "reconstruction_loss": 1.0,
    },
)

adata.obsm["X_corrected"] = model.transform(adata.obsm["X_pca"], adata.obs["batch"]).numpy()

sc.pp.neighbors(adata, use_rep="X_corrected", key_added="neighbors_corrected")
sc.tl.umap(adata, neighbors_key="neighbors_corrected", key_added="umap_corrected")
sc.pl.embedding(adata, basis="umap_corrected", color=["batch"])
```

`batch_metadata` accepts more than one column (e.g. `adata.obs[["batch", "donor"]]`) to correct
for several effects jointly — BIBOLA trains and averages one model per effect.

For small/in-memory datasets where chunking isn't needed, swap `ChunkIntegration` for
`BaseIntegration` (same constructor and `fit()`/`transform()` signature, full-batch training).

Set `fit(..., print_every=1)` for a per-epoch loss printout, or `fit(..., verbose=True)` to also
print the loss breakdown at every `run_epoch()` call (every chunk, for `ChunkIntegration`). The
full per-epoch loss history is available afterwards as `model.loss_history` (a `pandas.DataFrame`).

For a synthetic, fully self-contained walkthrough (data generation, training, t-SNE before/after),
see `workflow_with_simulated_data.ipynb`. For real single-cell benchmarks against Harmony,
see `pancreas_benchmark.ipynb` and `lung_atlas_benchmark.ipynb` (datasets in `data/`).

---

## Development

```bash
conda activate bibola
python -m pytest tests/ -q
```

Tests cover the individual building blocks (`bibola/spatial_autocorrelation.py`, `bibola/knn.py`,
`bibola/topology.py`, `bibola/MLP.py`); the integration workflow itself (`BaseIntegration`,
`ChunkIntegration`) is exercised end-to-end in the benchmark notebooks above.
