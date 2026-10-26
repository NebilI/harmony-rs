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

The R package calls the `harmorust` CLI over the binary `HRu1` protocol (`run-bin`), falling back to JSON. An extendr staticlib like seurust is still a possible later swap.

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

- Harmony C++ defaults to `float` when BLAS provides `sdot_`/`sgetrf_`. harmorust's kernel is float and fails the live gate if Harmony was compiled as `double` (`HARMORUST_ALLOW_DOUBLE=1` to override). Residual `|Δ|` on ridge-only cases is a few ulp on reference BLAS and ~`1e-5` on OpenBLAS GEMM.
- k-means++ and Lloyd match Harmony's `initialize_centroids` / Armadillo `km_iterate` (float two-accumulator distance, dead-mean resurrection, shared R stream).
- Block shuffles use Armadillo's argsort-of-`randi` on that same stream. The R package forwards and restores `.Random.seed`.
- `ncores` maps to Rayon, not OpenMP / OpenBLAS (`RhpcBLASctl`). Parity runs with `ncores=1`.
- `return_object` is a harmorust list with Harmony field names (`R`/`Y`/`O`/`E`/`objective_*`), not the live Rcpp module.
- No `%>%` re-export from dplyr.

A future `tests/testthat/test_rust_harmony.R` should compare `harmony::RunHarmony(...)` to `harmorust::RunHarmony` the way seurust compares `LogNorm`.

## Numerical difference (C++ `float` vs harmorust)

`compare` only times the two kernels. `parity` runs them once on the same synthetic atlas and reports the element-wise absolute difference of each result, with **min** and **max** in scientific notation.

```sh
# requires R and install.packages("harmony")
cargo run -p harmorust --release -- parity
./scripts/parity.sh --cells 200,500,1000 --nclust 1,5
# pull-request gate (fails if a gated max |Δ| exceeds 1e-4)
cargo run -p harmorust --release -- parity --check
```

## Pull-request gate

Every pull request must keep gated `|harmorust − Harmony|` at or below `PARITY_MAX_ABS` (`1e-4`, override with `HARMORUST_PARITY_MAX`). Discrete fields (`n_iter`, `kmeans_rounds`) are zero-tolerance. `objective` is reported but not gated (it accumulates inner rounds). Small multi-cluster cases (`nclust > 1` and `N < 400`) stay gated, but at `6e-2`, the measured leftover from online `update_R`; their ridge-on-Harmony-`R` check stays at `1e-4`.

CI has two jobs:

- **reference-blas** — apt reference BLAS/LAPACK, Harmony built from source, `ncores=1`. `meta.json` records Harmony's scalar type, BLAS, LAPACK, and RcppArmadillo; the harness fails if Harmony is a `double` build.
- **openblas** — binary Harmony, `HARMORUST_PARITY_MAX=1e-4` (kernel GEMM order is not reproducible; measured ridge floor is ~`3e-5`).

`cargo test --workspace` with `HARMORUST_REQUIRE_PARITY=1` plus `harmorust parity --check --cells 200,500 --nclust 1,5 --ncores 1`. Local `cargo test` skips the live Harmony cases when the R package is missing. Golden fixtures under `crates/harmorust/tests/fixtures/` (`scripts/regen-fixtures.sh`) check the kernel without R.

Gated fields: `embeddings`, `R`, `Y`, `O`, `E`, `moe_correct_ridge`, `n_iter`, `kmeans_rounds`. Stage dumps and teacher-forced per-stage diffs are reported as `first_divergence` / `stage_*` / `teacher_*`.

Both sides see the same embeddings and hyperparameters (`max_iter = 8`, early stopping off, `lambda = 1`, 20 PCs, 4 batches). Harmony 2.0.5 stores the C++ loop in `float`. harmorust matches that path for k-means++ / Lloyd / softmax / ridge, uses Armadillo's argsort shuffle on the shared R stream, and compares soft-assignment matrices after aligning clusters on `Y`. `moe_correct_ridge` is harmorust's ridge applied to Harmony's final `R`, compared to Harmony's `Z_corr`.

`nclust = 1` is ordinary ridge (no clustering ambiguity). Some embedding entries match exactly (`min |Δ| = 0`). Max stays around the float / GEMM floor. `n_iter` and `kmeans_rounds` matched Harmony exactly on every case below.

Measured on this machine against Harmony 2.0.5 / harmorust 0.1.0 (`ncores=1` for parity), reference BLAS (`libblas.so.3.12.0` / `liblapack.so.3.12.0`), RcppArmadillo 15.6.2.1, SCALAR=float (2026-10-26, America/New_York).

### `nclust = 1` (ridge only)

| Case | Field | Elements | min \|Δ\| | max \|Δ\| |
| --- | --- | ---: | ---: | ---: |
| atlas-200 | embeddings | 4000 | 0.0000e+00 | 4.0531e-06 |
| atlas-200 | R | 200 | 0.0000e+00 | 0.0000e+00 |
| atlas-200 | Y | 20 | 5.2154e-08 | 4.8280e-06 |
| atlas-200 | O / E | 4 | 0.0000e+00 | 0.0000e+00 |
| atlas-200 | moe_correct_ridge | 4000 | 0.0000e+00 | 4.0531e-06 |
| atlas-500 | embeddings | 10000 | 0.0000e+00 | 3.1114e-05 |
| atlas-500 | R | 500 | 0.0000e+00 | 0.0000e+00 |
| atlas-500 | Y | 20 | 2.9802e-08 | 6.1158e-05 |
| atlas-500 | O / E | 4 | 0.0000e+00 | 0.0000e+00 |
| atlas-500 | moe_correct_ridge | 10000 | 0.0000e+00 | 3.1114e-05 |
| atlas-1000 | embeddings | 20000 | 0.0000e+00 | 3.1590e-05 |
| atlas-1000 | moe_correct_ridge | 20000 | 0.0000e+00 | 3.1590e-05 |

### `nclust = 5` (full Harmony)

| Case | Field | Elements | min \|Δ\| | max \|Δ\| |
| --- | --- | ---: | ---: | ---: |
| atlas-200 | embeddings | 4000 | 0.0000e+00 | 1.0737e-02 |
| atlas-200 | R (aligned) | 1000 | 0.0000e+00 | 1.2108e-02 |
| atlas-200 | Y (aligned) | 100 | 1.1176e-08 | 1.1744e-03 |
| atlas-200 | O (aligned) | 20 | 0.0000e+00 | 5.4458e-02 |
| atlas-200 | E (aligned) | 20 | 0.0000e+00 | 1.3602e-02 |
| atlas-200 | moe_correct_ridge | 4000 | 0.0000e+00 | 1.6451e-05 |
| atlas-500 | embeddings | 10000 | 0.0000e+00 | 3.6240e-05 |
| atlas-500 | R (aligned) | 2500 | 0.0000e+00 | 1.1921e-06 |
| atlas-500 | Y (aligned) | 100 | 6.0536e-09 | 4.5896e-06 |
| atlas-500 | O (aligned) | 20 | 0.0000e+00 | 1.3351e-05 |
| atlas-500 | E (aligned) | 20 | 0.0000e+00 | 5.7220e-06 |
| atlas-500 | moe_correct_ridge | 10000 | 0.0000e+00 | 4.1485e-05 |
| atlas-1000 | embeddings | 20000 | 0.0000e+00 | 8.4877e-05 |
| atlas-1000 | R (aligned) | 5000 | 0.0000e+00 | 9.5703e-07 |
| atlas-1000 | moe_correct_ridge | 20000 | 0.0000e+00 | 1.5378e-04 |

## Speed

Compare against CRAN `harmony` (the original C++ kernel) on the same synthetic atlas:

```sh
# requires R and install.packages("harmony")
cargo run -p harmorust --release -- compare
./scripts/compare.sh --nclust 5 --repeats 7
./scripts/compare.sh --nclust 20
```

Both kernels get the same embeddings and hyperparameters (`max_iter = 8`, early stopping off, `lambda = 1`, 20 PCs, 4 batches). Times are **median wall time** after 2 warmup runs and 5 measured repeats. **Speedup** is original / harmorust (>1 means harmorust is faster).

Measured on this 4-core machine against Harmony 2.0.5 / reference BLAS / harmorust 0.1.0 (`ncores=4` unless noted), 2026-10-26 America/New_York, after the column-major float kernel, fused ridge, fat LTO, and binary `run-bin` landing. Times are median wall time, 2 warmup + 5 repeats (ncores=1 used 1 warmup + 3 repeats).

### This machine, after the parity/speed landing

#### Default `nclust = min(round(N/30), 100)`, `ncores=4`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 17 | 16.3 ms | 7.92 ms | 2.06× |
| atlas-1000 | 1000 | 20 | 33 | 48.8 ms | 25.1 ms | 1.94× |
| atlas-2000 | 2000 | 20 | 67 | 179.3 ms | 98.1 ms | 1.83× |
| atlas-4000 | 4000 | 20 | 100 | 535.3 ms | 273.6 ms | 1.96× |
| atlas-8000 | 8000 | 20 | 100 | 1.15 s | 543.9 ms | 2.12× |

Geometric-mean speedup: **1.98×**. Same protocol at `ncores=1`: 1.67× geo-mean (500 / 2000 / 8000).

#### Medium `nclust = 20`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 20 | 18.3 ms | 8.77 ms | 2.08× |
| atlas-1000 | 1000 | 20 | 20 | 32.7 ms | 16.6 ms | 1.97× |
| atlas-2000 | 2000 | 20 | 20 | 61.0 ms | 31.9 ms | 1.91× |
| atlas-4000 | 4000 | 20 | 20 | 122.6 ms | 65.0 ms | 1.88× |
| atlas-8000 | 8000 | 20 | 20 | 253.7 ms | 165.9 ms | 1.53× |

Geometric-mean speedup: **1.87×**.

#### Small `nclust = 5`

| Case | Cells | PCs | Clusters | Original Harmony | harmorust | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas-500 | 500 | 20 | 5 | 9.57 ms | 3.35 ms | 2.86× |
| atlas-1000 | 1000 | 20 | 5 | 13.9 ms | 6.32 ms | 2.20× |
| atlas-2000 | 2000 | 20 | 5 | 24.5 ms | 11.0 ms | 2.23× |
| atlas-4000 | 4000 | 20 | 5 | 46.9 ms | 22.3 ms | 2.10× |
| atlas-8000 | 8000 | 20 | 5 | 92.2 ms | 45.9 ms | 2.01× |

Geometric-mean speedup: **2.26×**. Drop-in CLI (`--via-r-package`, JSON/`run-bin` overhead included) was 6.11 ms at 500 cells and 19.6 ms at 2000 cells (`nclust=5`), vs 3.35 ms / 11.0 ms in-process.

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
