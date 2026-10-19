# Parity with immunogenomics/harmony 2.0.4 / 2.0.5

harmorust reimplements the published Harmony2 loop, not a copy of the GPL-3 C++ sources. The public surface is meant to be a drop-in for `RunHarmony()`.

## Matched

| Harmony | harmorust |
| --- | --- |
| `RunHarmony(matrix, meta, vars_use)` | `run_harmony` / `run_harmony_multi` + `r/harmorust` |
| `RunHarmony.Seurat` | `RunHarmony.Seurat` |
| `RunHarmony.SingleCellExperiment` | `RunHarmony.SingleCellExperiment` |
| Embeddings returned cells × PCs | `HarmonyResult.embeddings` |
| `theta`, `sigma`, `lambda`, `nclust`, `max_iter`, `early_stop`, `verbose` | same |
| `harmony_options()` (`alpha`, `tau`, `block.size`, `epsilon.*`, `batch.prop.cutoff`) | same fields |
| `lambda = NULL` automatic ridge | `lambda: None` → `α · E` |
| `theta` scaled by `1 - exp(-(N_b / (K τ))²)` | `expand_theta` |
| Cosine soft k-means + online block `R` updates | `cluster.rs` |
| Diversity penalty `((2E+1)/(O+E+1))^θ` | same |
| Per-cluster MoE ridge from `Z_orig`, intercept kept | `correct.rs` |
| Arrowhead inverse when one covariate | `invert_arrowhead` |
| Skip a batch in a cluster below `batch.prop.cutoff` | same |
| Skip a covariate with only one qualifying level | same |
| `return_object` fields `R`, `Y`, `O`, objectives | `harmorust_harmony` list (`R`/`Y`/`O`/`objective_*`) |
| `plot_convergence` / convergence plot | `HarmonyConvergencePlot()` |
| Legacy arg rejection (`do_pca`, `max.iter.harmony`, …) | `.check_legacy_args` |
| Seurat `group.by.vars`, `reduction.use`, `reduction.save` | `r/harmorust` |
| Two or more covariates | stacked one-hot `Design` |
| `ncores` | `RAYON_NUM_THREADS` for the Rust CLI child |

Harmony 2.0.5 only **exports** `RunHarmony`, `harmony_options`, and `%>%`. harmorust matches those entry points (minus the dplyr pipe re-export) and also exports `HarmonyConvergencePlot` so `plot_convergence=TRUE` / `return_object=TRUE` workflows work without reaching into internals.

The R package calls the `harmorust` CLI over JSON (`HARMORUST_BIN` or `PATH`). That is the current bind path; an extendr staticlib like seurust is still a possible later swap.

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
- `ncores` maps to Rayon, not OpenMP / OpenBLAS (`RhpcBLASctl`).
- `return_object` is a harmorust list with Harmony field names, not the Rcpp C++ module (no live `cluster_cpp` / `moe_correct_ridge_cpp` methods).
- No `%>%` re-export from dplyr.

A future `tests/testthat/test_rust_harmony.R` should compare `harmony::RunHarmony(...)` to `harmorust::RunHarmony` the way seurust compares `LogNorm`.

## Speed

Compare against CRAN `harmony` (the original C++ kernel) on the same synthetic atlas:

```sh
# requires R and install.packages("harmony")
cargo run -p harmorust --release -- compare
./scripts/compare.sh --nclust 5 --repeats 7
```

Both kernels get the same embeddings and hyperparameters (`max_iter = 8`, early stopping off, `lambda = 1`, 20 PCs, 4 batches). Times are **median wall time** after 2 warmup runs and 5 measured repeats. **Speedup** is original / harmorust (>1 means harmorust is faster).

Measured on a 4-core Intel Xeon, OpenBLAS 0.3.26, Harmony 2.0.5, harmorust 0.1.0 (`ncores=4`), after the `update_r` cache, incremental k-means++, the objective log table, and ridge fits that walk per-batch cell lists.

### Default `nclust = min(round(N/30), 100)`

This is Harmony's usual workload (many soft clusters, one ridge fit each).

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 17 | 18.1 ms | 7.57 ms | 2.40× |
| atlas-1000 | 1000 | 20 | 33 | 54.0 ms | 25.5 ms | 2.12× |
| atlas-2000 | 2000 | 20 | 67 | 173.2 ms | 110.1 ms | 1.57× |
| atlas-4000 | 4000 | 20 | 100 | 527.6 ms | 334.5 ms | 1.58× |
| atlas-8000 | 8000 | 20 | 100 | 1.15 s | 804.0 ms | 1.44× |

Geometric-mean speedup: **1.78×** (was 1.56× after the objective table). At 8k cells harmorust went 877.5 ms → 804.0 ms.

### Small `nclust = 5` (matched to planted cell types)

Same atlas as `harmorust bench`, with `nclust` pinned to biology rather than Harmony's default.

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 5 | 8.88 ms | 2.30 ms | 3.87× |
| atlas-1000 | 1000 | 20 | 5 | 14.3 ms | 4.82 ms | 2.96× |
| atlas-2000 | 2000 | 20 | 5 | 24.5 ms | 9.52 ms | 2.57× |
| atlas-4000 | 4000 | 20 | 5 | 47.0 ms | 17.9 ms | 2.62× |
| atlas-8000 | 8000 | 20 | 5 | 90.8 ms | 34.9 ms | 2.60× |

Geometric-mean speedup: **2.89×** (was 2.58×).

The C++ hot path is already one file and is linked to OpenBLAS. Remaining cost is mostly memory layout and per-cell loops that are still row-strided.

`cargo run -p harmorust --release -- bench --cells 4000` still prints a single harmorust wall time.
