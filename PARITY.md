# Parity with immunogenomics/harmony 2.0.4 / 2.0.5

harmorust reimplements the published Harmony2 loop, not a copy of the GPL-3 C++ sources. The public surface is meant to be a drop-in for `RunHarmony()`.

## Matched

| Harmony | harmorust |
| --- | --- |
| `RunHarmony(matrix, meta, vars_use)` | `run_harmony` / `run_harmony_multi` |
| Embeddings returned cells × PCs | `HarmonyResult.embeddings` |
| `theta`, `sigma`, `lambda`, `nclust`, `max_iter` | `HarmonyConfig` |
| `harmony_options()` (`alpha`, `tau`, `block.size`, `epsilon.*`, `batch.prop.cutoff`) | same fields |
| `lambda = NULL` automatic ridge | `lambda: None` → `α · E` |
| `theta` scaled by `1 - exp(-(N_b / (K τ))²)` | `expand_theta` |
| Cosine soft k-means + online block `R` updates | `cluster.rs` |
| Diversity penalty `((2E+1)/(O+E+1))^θ` | same |
| Per-cluster MoE ridge from `Z_orig`, intercept kept | `correct.rs` |
| Arrowhead inverse when one covariate | `invert_arrowhead` |
| Skip a batch in a cluster below `batch.prop.cutoff` | same |
| Skip a covariate with only one qualifying level | same |
| `return_object` fields `R`, `Y`, `O` | `HarmonyResult.{r,y,o}` |
| Seurat `group.by.vars`, `reduction.use`, `reduction.save` | `r/harmorust` |
| Two or more covariates | stacked one-hot `Design` |

seurust is **not** a Harmony implementation. It supplies the Seurat kernels that produce the PCA embedding Harmony consumes (`NormalizeData`, `ScaleData`, SNN). Compose them:

```r
# seurust-backed preprocessing
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj)
obj <- RunPCA(obj)
# harmorust kernel
obj <- harmorust::RunHarmony(obj, group.by.vars = "donor")
```

## Not bit-identical (yet)

- Harmony C++ defaults to `float`; we use `f64`.
- k-means++ uses the same sampling idea but not Armadillo's `arma::kmeans`.
- Block shuffles are seeded independently of R's `.Random.seed`.
- No OpenMP / OpenBLAS thread control (`ncores`); Rayon uses the crate default pool.
- R / Seurat / SingleCellExperiment wrappers are API-complete in source; the extendr bind still needs `cargo` at install time, same as seurust.

A future `tests/testthat/test_rust_harmony.R` should compare `harmony::RunHarmony(..., seed=…)` to `harmorust::RunHarmony` the way seurust compares `LogNorm`. That needs R + the CRAN Harmony package on the machine.

## Speed

Compare against CRAN `harmony` (the original C++ kernel) on the same synthetic atlas:

```sh
# requires R and install.packages("harmony")
cargo run -p harmorust --release -- compare
./scripts/compare.sh --nclust 5 --repeats 7
```

Both kernels get the same embeddings and hyperparameters (`max_iter = 8`, early stopping off, `lambda = 1`, 20 PCs, 4 batches). Times are **median wall time** after 2 warmup runs and 5 measured repeats. **Speedup** is original / harmorust (>1 means harmorust is faster).

Measured on a 4-core Intel Xeon, OpenBLAS 0.3.26, Harmony 2.0.5, harmorust 0.1.0 (`ncores=4`):

### Default `nclust = min(round(N/30), 100)`

This is Harmony's usual workload (many soft clusters, one ridge fit each).

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 17 | 15.7 ms | 18.4 ms | 0.85× |
| atlas-1000 | 1000 | 20 | 33 | 56.5 ms | 68.8 ms | 0.82× |
| atlas-2000 | 2000 | 20 | 67 | 171.1 ms | 292.6 ms | 0.58× |
| atlas-4000 | 4000 | 20 | 100 | 515.2 ms | 942.8 ms | 0.55× |
| atlas-8000 | 8000 | 20 | 100 | 1.12 s | 1.96 s | 0.57× |

Geometric-mean speedup: **0.66×** (original C++ is faster here). Repeat MAD was under 2% of the median on both sides.

### Small `nclust = 5` (matched to planted cell types)

Same atlas as `harmorust bench`, with `nclust` pinned to biology rather than Harmony's default.

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 5 | 8.24 ms | 5.63 ms | 1.46× |
| atlas-1000 | 1000 | 20 | 5 | 13.2 ms | 11.7 ms | 1.13× |
| atlas-2000 | 2000 | 20 | 5 | 23.7 ms | 22.5 ms | 1.05× |
| atlas-4000 | 4000 | 20 | 5 | 44.8 ms | 44.1 ms | 1.02× |
| atlas-8000 | 8000 | 20 | 5 | 89.0 ms | 87.6 ms | 1.02× |

Geometric-mean speedup: **1.12×**. At this small K the kernels are essentially tied above ~2k cells.

The C++ hot path is already one file and is linked to OpenBLAS. harmorust still has room on the default-`nclust` path (per-cluster ridge in f64, less BLAS-heavy GEMM than Armadillo). Intended Rust advantages, not yet a win at K≈100:

- Rayon over independent per-cluster ridge fits
- `ndarray` GEMM for cosine distances (`YᵀZ`)
- O(covariates) diversity updates instead of scanning every one-hot row
- No Rcpp / Armadillo copy at the R boundary once extendr is wired

`cargo run -p harmorust --release -- bench --cells 4000` still prints a single harmorust wall time.
