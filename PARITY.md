# Parity with immunogenomics/harmony 2.0.4

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

The C++ hot path is already one file. Rust wins by:

- Rayon over independent per-cluster ridge fits
- `ndarray` GEMM for cosine distances (`YᵀZ`)
- O(covariates) diversity updates instead of scanning every one-hot row
- No Rcpp / Armadillo copy at the R boundary once extendr is wired

Compare against CRAN `harmony` (the original C++ kernel) on the same synthetic atlas:

```sh
# requires R and install.packages("harmony")
cargo run -p harmorust --release -- compare
```

The command prints a table of median wall times and `original / harmorust` speedup. Protocol: planted atlas, 20 PCs, 4 batches, 5 types, Harmony's default `nclust = min(round(N/30), 100)`, `max_iter = 8`, early stopping off, `lambda = 1`, warmup then several measured repeats. `--cells`, `--nclust`, `--repeats`, and `--warmup` are flags so the same harness can be re-run.

`cargo run -p harmorust --release -- bench --cells 4000` still prints a single harmorust wall time.
