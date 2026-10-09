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

Measured on a 4-core Intel Xeon, OpenBLAS 0.3.26, Harmony 2.0.5, harmorust 0.1.0 (`ncores=4`), after the `update_r` cache, incremental k-means++ seeding, and the cluster × batch objective table.

### Default `nclust = min(round(N/30), 100)`

This is Harmony's usual workload (many soft clusters, one ridge fit each).

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 17 | 17.0 ms | 8.07 ms | 2.11× |
| atlas-1000 | 1000 | 20 | 33 | 50.0 ms | 30.7 ms | 1.63× |
| atlas-2000 | 2000 | 20 | 67 | 188.2 ms | 125.4 ms | 1.50× |
| atlas-4000 | 4000 | 20 | 100 | 532.5 ms | 379.2 ms | 1.40× |
| atlas-8000 | 8000 | 20 | 100 | 1.13 s | 877.5 ms | 1.29× |

Geometric-mean speedup: **1.56×** (was 1.21× after k-means++). At 8k cells harmorust went 1.07 s → 877.5 ms.

### Small `nclust = 5` (matched to planted cell types)

Same atlas as `harmorust bench`, with `nclust` pinned to biology rather than Harmony's default.

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 5 | 9.19 ms | 2.87 ms | 3.20× |
| atlas-1000 | 1000 | 20 | 5 | 14.0 ms | 5.08 ms | 2.75× |
| atlas-2000 | 2000 | 20 | 5 | 24.6 ms | 10.2 ms | 2.41× |
| atlas-4000 | 4000 | 20 | 5 | 46.7 ms | 20.2 ms | 2.31× |
| atlas-8000 | 8000 | 20 | 5 | 90.9 ms | 39.1 ms | 2.33× |

Geometric-mean speedup: **2.58×** (was 2.04×).

The C++ hot path is already one file and is linked to OpenBLAS. Remaining cost is mostly per-cluster ridge fits.

`cargo run -p harmorust --release -- bench --cells 4000` still prints a single harmorust wall time.
