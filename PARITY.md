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

## Numerical difference (C++ `float` vs Rust `f64`)

`compare` only times the two kernels. `parity` runs them once on the same synthetic atlas and reports the element-wise absolute difference of each result, with **min** and **max** in scientific notation.

```sh
# requires R and install.packages("harmony")
cargo run -p harmorust --release -- parity
./scripts/parity.sh --cells 200,500,1000 --nclust 1,5
```

Both sides see the same embeddings and hyperparameters (`max_iter = 8`, early stopping off, `lambda = 1`, 20 PCs, 4 batches). Harmony 2.0.5 stores the C++ loop in `float`; harmorust uses `f64`. Soft-assignment matrices `R`, `Y`, and `O` are compared after aligning clusters on `Y` (k-means++ can permute labels). `moe_correct_ridge` is harmorust's ridge applied to Harmony's final `R`, compared to Harmony's `Z_corr` — that isolates the correction from clustering.

`nclust = 1` is ordinary ridge (no clustering ambiguity), so those rows are the cleanest float-vs-f64 signal. Larger `nclust` also includes independent k-means++ seeding and block shuffles, so max `|Δ|` can jump from machine-epsilon-for-float (~1e-7) to O(1).

Measured on this machine against Harmony 2.0.5 / harmorust 0.1.0 (`ncores=4`). Re-run `./scripts/parity.sh` to refresh the table.

### `nclust = 1` (ridge only)

One soft cluster, so k-means++ cannot permute labels. Embedding `|Δ|` stays in the float-ulp range (min ~1e-10, max ~3e-5). `R` and `O` match to 0 or ~1e-16. The objective is a sum, so small per-cell errors accumulate.

| Case | Field | Elements | min \|Δ\| | max \|Δ\| |
| --- | --- | ---: | ---: | ---: |
| atlas-200 | embeddings | 4000 | 4.9781e-09 | 5.3873e-06 |
| atlas-200 | R | 200 | 0.0000e+00 | 1.1102e-16 |
| atlas-200 | Y | 20 | 9.8149e-08 | 5.4073e-06 |
| atlas-200 | O | 4 | 0.0000e+00 | 0.0000e+00 |
| atlas-200 | moe_correct_ridge | 4000 | 3.3742e-08 | 5.2837e-06 |
| atlas-500 | embeddings | 10000 | 2.8591e-09 | 3.0002e-05 |
| atlas-500 | R | 500 | 0.0000e+00 | 1.1102e-16 |
| atlas-500 | Y | 20 | 1.4428e-07 | 5.2817e-05 |
| atlas-500 | O | 4 | 0.0000e+00 | 0.0000e+00 |
| atlas-500 | moe_correct_ridge | 10000 | 2.1233e-09 | 2.9950e-05 |
| atlas-1000 | embeddings | 20000 | 1.1650e-10 | 3.0109e-05 |
| atlas-1000 | R | 1000 | 0.0000e+00 | 0.0000e+00 |
| atlas-1000 | Y | 20 | 1.3447e-07 | 4.7914e-05 |
| atlas-1000 | O | 4 | 0.0000e+00 | 0.0000e+00 |
| atlas-1000 | moe_correct_ridge | 20000 | 6.2431e-09 | 3.0097e-05 |

### `nclust = 5` (full Harmony)

Same atlas with five soft clusters. When k-means++ lands on the same partition (500 / 1000 cells) the corrected embeddings stay near float ulp. On the 200-cell atlas the partitions diverged (`R` max \|Δ\| ≈ 1), so end-to-end embeddings differ by O(1) — but `moe_correct_ridge` on Harmony's own `R` is still ~1e-5, which is the float-vs-f64 gap of the ridge itself.

| Case | Field | Elements | min \|Δ\| | max \|Δ\| |
| --- | --- | ---: | ---: | ---: |
| atlas-200 | embeddings | 4000 | 2.2896e-05 | 1.8171e+00 |
| atlas-200 | R (aligned) | 1000 | 3.3260e-18 | 9.9876e-01 |
| atlas-200 | Y (aligned) | 100 | 1.7138e-05 | 2.5378e-01 |
| atlas-200 | O (aligned) | 20 | 1.7079e-05 | 9.6200e+00 |
| atlas-200 | moe_correct_ridge | 4000 | 1.7635e-10 | 9.5990e-06 |
| atlas-500 | embeddings | 10000 | 2.5231e-12 | 3.9461e-05 |
| atlas-500 | R (aligned) | 2500 | 1.5188e-20 | 3.2558e-06 |
| atlas-500 | Y (aligned) | 100 | 1.0128e-09 | 3.8961e-06 |
| atlas-500 | O (aligned) | 20 | 2.7621e-07 | 1.8340e-05 |
| atlas-500 | moe_correct_ridge | 10000 | 9.9382e-11 | 4.1187e-05 |
| atlas-1000 | embeddings | 20000 | 1.8192e-10 | 1.3783e-04 |
| atlas-1000 | R (aligned) | 5000 | 1.9400e-20 | 1.7757e-06 |
| atlas-1000 | Y (aligned) | 100 | 4.7780e-10 | 1.6509e-05 |
| atlas-1000 | O (aligned) | 20 | 1.5460e-06 | 4.9945e-05 |
| atlas-1000 | moe_correct_ridge | 20000 | 3.3771e-10 | 1.2222e-04 |

## Speed

Compare against CRAN `harmony` (the original C++ kernel) on the same synthetic atlas:

```sh
# requires R and install.packages("harmony")
cargo run -p harmorust --release -- compare
./scripts/compare.sh --nclust 5 --repeats 7
./scripts/compare.sh --nclust 20
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

### Medium `nclust = 20`

Same cell-count list with soft clusters fixed at 20 — between the biology-matched small-K case and Harmony's default.

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 20 | 18.4 ms | 8.36 ms | 2.20× |
| atlas-1000 | 1000 | 20 | 20 | 31.1 ms | 15.6 ms | 2.00× |
| atlas-2000 | 2000 | 20 | 20 | 57.9 ms | 36.2 ms | 1.60× |
| atlas-4000 | 4000 | 20 | 20 | 117.6 ms | 65.0 ms | 1.81× |
| atlas-8000 | 8000 | 20 | 20 | 252.5 ms | 134.7 ms | 1.87× |

Geometric-mean speedup: **1.89×**. Reproducible with `./scripts/compare.sh --nclust 20`.

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
