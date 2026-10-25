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

### This machine, before vectorized `exp`/`ln`

Same protocol as the tables below (`max_iter=8`, early stopping off, `lambda=1`, 20 PCs, 4 batches, 2 warmup + 5 repeats, `ncores=4`). Recorded immediately before switching softmax and the objective entropy term to a vectorized `exp`/`ln`. Use these rust columns as the before/after baseline for that change — original Harmony times on this box also drift slightly from the older published table.

#### Default `nclust = min(round(N/30), 100)`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 17 | 15.8 ms | 6.94 ms | 2.28× |
| atlas-1000 | 1000 | 20 | 33 | 45.9 ms | 25.0 ms | 1.84× |
| atlas-2000 | 2000 | 20 | 67 | 166.7 ms | 108.7 ms | 1.53× |
| atlas-4000 | 4000 | 20 | 100 | 509.6 ms | 329.2 ms | 1.55× |
| atlas-8000 | 8000 | 20 | 100 | 1.06 s | 778.2 ms | 1.37× |

Geometric-mean speedup: **1.68×**.

#### Medium `nclust = 20`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 20 | 17.5 ms | 7.88 ms | 2.22× |
| atlas-1000 | 1000 | 20 | 20 | 30.6 ms | 16.0 ms | 1.92× |
| atlas-2000 | 2000 | 20 | 20 | 57.0 ms | 30.4 ms | 1.87× |
| atlas-4000 | 4000 | 20 | 20 | 115.2 ms | 61.5 ms | 1.87× |
| atlas-8000 | 8000 | 20 | 20 | 235.0 ms | 133.5 ms | 1.76× |

Geometric-mean speedup: **1.92×**.

#### Small `nclust = 5`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 5 | 8.19 ms | 2.22 ms | 3.69× |
| atlas-1000 | 1000 | 20 | 5 | 13.3 ms | 4.57 ms | 2.92× |
| atlas-2000 | 2000 | 20 | 5 | 23.8 ms | 8.93 ms | 2.66× |
| atlas-4000 | 4000 | 20 | 5 | 47.2 ms | 19.3 ms | 2.44× |
| atlas-8000 | 8000 | 20 | 5 | 98.1 ms | 35.9 ms | 2.73× |

Geometric-mean speedup: **2.86×**.

### This machine, after vectorized `exp`/`ln` (glibc libmvec via C)

Same protocol after softmax / objective entropy call glibc libmvec over contiguous `K × N` rows (AVX-512 8-wide when present, else AVX2). That landing needed a C shim because stable Rust cannot pass SIMD types through FFI. **This-change** speedup is harmorust-before / harmorust-after on this box. The next section replaces the C shim with a Rust `std::arch` polynomial (rustc ≥ 1.89) and remeasures.

Default `nclust` geo-mean vs original: **1.68× → 1.99×**. At 8k cells harmorust went **778.2 ms → 692.7 ms** (1.12× on our side).

#### Default `nclust` after

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | vs original | vs this-machine before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 17 | 15.4 ms | 5.52 ms | 2.80× | 1.26× |
| atlas-1000 | 1000 | 20 | 33 | 46.4 ms | 21.0 ms | 2.21× | 1.19× |
| atlas-2000 | 2000 | 20 | 67 | 164.6 ms | 89.0 ms | 1.85× | 1.22× |
| atlas-4000 | 4000 | 20 | 100 | 499.4 ms | 286.6 ms | 1.74× | 1.15× |
| atlas-8000 | 8000 | 20 | 100 | 1.08 s | 692.7 ms | 1.55× | 1.12× |

Geometric-mean vs original: **1.99×**. Geometric-mean vs this-machine before: **1.19×**.

#### Medium `nclust = 20` after

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | vs original | vs this-machine before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 20 | 17.7 ms | 6.94 ms | 2.55× | 1.14× |
| atlas-1000 | 1000 | 20 | 20 | 30.7 ms | 13.9 ms | 2.21× | 1.15× |
| atlas-2000 | 2000 | 20 | 20 | 57.9 ms | 26.0 ms | 2.22× | 1.17× |
| atlas-4000 | 4000 | 20 | 20 | 114.0 ms | 52.2 ms | 2.18× | 1.18× |
| atlas-8000 | 8000 | 20 | 20 | 232.8 ms | 115.6 ms | 2.01× | 1.15× |

Geometric-mean vs original: **2.23×**. Geometric-mean vs this-machine before: **1.16×**.

#### Small `nclust = 5` after

First 5-repeat pass had a noisy 4k-cell rust median (33.5 ms). The table below is a 7-repeat re-run on the same binary (4k settled at 17.4 ms).

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | vs original | vs this-machine before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 5 | 8.38 ms | 1.91 ms | 4.40× | 1.16× |
| atlas-1000 | 1000 | 20 | 5 | 13.7 ms | 3.95 ms | 3.47× | 1.16× |
| atlas-2000 | 2000 | 20 | 5 | 24.2 ms | 7.32 ms | 3.31× | 1.22× |
| atlas-4000 | 4000 | 20 | 5 | 45.4 ms | 17.4 ms | 2.61× | 1.11× |
| atlas-8000 | 8000 | 20 | 5 | 88.8 ms | 31.8 ms | 2.80× | 1.13× |

Geometric-mean vs original: **3.26×**. Geometric-mean vs this-machine before: **1.16×**.

### This machine, after `std::arch` exp/ln (rustc 1.89)

Same protocol after dropping the C shim. Softmax / objective entropy use a Rust `std::arch` AVX-512 (else AVX2+FMA) Cephes/fdlibm polynomial over contiguous `K × N` rows. **This-change** is still harmorust-before (scalar) / harmorust-after. Versus the libmvec landing the kernel is a few percent slower.

Default `nclust` geo-mean vs original: **1.68× → 1.94×**. At 8k cells harmorust went **778.2 ms → 738.5 ms** (1.05× on our side; libmvec was 692.7 ms).

#### Default `nclust` after `std::arch`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | vs original | vs this-machine before | vs libmvec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 17 | 16.0 ms | 5.74 ms | 2.79× | 1.21× | 0.96× |
| atlas-1000 | 1000 | 20 | 33 | 46.0 ms | 20.5 ms | 2.24× | 1.22× | 1.02× |
| atlas-2000 | 2000 | 20 | 67 | 165.0 ms | 90.7 ms | 1.82× | 1.20× | 0.98× |
| atlas-4000 | 4000 | 20 | 100 | 501.9 ms | 307.8 ms | 1.63× | 1.07× | 0.93× |
| atlas-8000 | 8000 | 20 | 100 | 1.09 s | 738.5 ms | 1.48× | 1.05× | 0.94× |

Geometric-mean vs original: **1.94×**. Geometric-mean vs this-machine before: **1.15×**. Geometric-mean vs libmvec: **0.97×**.

#### Medium `nclust = 20` after `std::arch`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | vs original | vs this-machine before | vs libmvec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 20 | 17.4 ms | 6.90 ms | 2.52× | 1.14× | 1.01× |
| atlas-1000 | 1000 | 20 | 20 | 30.7 ms | 14.0 ms | 2.19× | 1.14× | 0.99× |
| atlas-2000 | 2000 | 20 | 20 | 57.1 ms | 25.9 ms | 2.21× | 1.17× | 1.00× |
| atlas-4000 | 4000 | 20 | 20 | 113.3 ms | 54.1 ms | 2.09× | 1.14× | 0.96× |
| atlas-8000 | 8000 | 20 | 20 | 232.5 ms | 121.9 ms | 1.91× | 1.10× | 0.95× |

Geometric-mean vs original: **2.17×**. Geometric-mean vs this-machine before: **1.14×**. Geometric-mean vs libmvec: **0.98×**.

#### Small `nclust = 5` after `std::arch`

5-repeat medians were stable this pass (4k MAD 0.30 ms), so no 7-repeat re-run.

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | vs original | vs this-machine before | vs libmvec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 5 | 8.27 ms | 2.02 ms | 4.08× | 1.10× | 0.95× |
| atlas-1000 | 1000 | 20 | 5 | 13.4 ms | 4.09 ms | 3.27× | 1.12× | 0.97× |
| atlas-2000 | 2000 | 20 | 5 | 23.9 ms | 8.01 ms | 2.98× | 1.11× | 0.91× |
| atlas-4000 | 4000 | 20 | 5 | 45.0 ms | 17.0 ms | 2.65× | 1.14× | 1.02× |
| atlas-8000 | 8000 | 20 | 5 | 89.1 ms | 31.5 ms | 2.82× | 1.14× | 1.01× |

Geometric-mean vs original: **3.12×**. Geometric-mean vs this-machine before: **1.12×**. Geometric-mean vs libmvec: **0.97×**.

### Published tables (previous campaign machine, before this change)

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

The C++ hot path is already one file and is linked to OpenBLAS. Softmax `exp` and objective entropy `ln` now use a Rust `std::arch` polynomial on contiguous rows (rustc ≥ 1.89; AVX-512 when present, else AVX2). Remaining cost is mostly memory layout in the other per-cell loops (assignments, ridge apply) that are still row-strided.

`cargo run -p harmorust --release -- bench --cells 4000` still prints a single harmorust wall time.
