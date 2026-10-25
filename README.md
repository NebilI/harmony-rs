# harmorust

Rust kernels for [Harmony](https://github.com/immunogenomics/harmony), built the same way [seurust](https://github.com/NebilI/seurust) ports Seurat.

Harmony is a good next target. Its hot path is one C++ file (`src/harmony.cpp`): cosine soft k-means, a diversity penalty on batch composition, and a per-cluster mixture-of-experts ridge. That loop only sees a PCA embedding and a batch vector — exactly what Seurat / seurust already produce. No new object model is required.

**It works.** On a planted two-donor atlas, `nclust` matched to the number of cell types, donor centroids collapse and cell-type geometry is unchanged:

```
batch separation  2.75 → 0.07
batch LISI        1.00 → 1.95   (2 donors → fully mixed)
type separation   3.21 → 3.21
```

`nclust = 1` reduces to ordinary ridge regression and removes a global batch shift (2.75 → 0.02), which is the diagnostic Harmony itself documents.

## What is in this repo

| Path | Role |
| --- | --- |
| `crates/harmorust` | Rust crate + CLI. This is the kernel. |
| `web` | Next.js demo: synthetic atlas, before/after PCs, LISI. |
| `r/harmorust` | R package with Harmony's public API (`RunHarmony` for matrix / Seurat / SCE, `harmony_options`, convergence plot). Calls the Rust CLI. |
| `examples/with_seurust.R` | How the two packages sit in one Seurat workflow. |

## Pipeline next to seurust

```text
counts
  → seurust   NormalizeData / ScaleData / FindNeighbors kernels
  → Seurat    RunPCA
  → harmorust RunHarmony(embeddings, donor)
  → seurust   SNN / clustering on the harmony reduction
```

Harmony never touches the count matrix. If seurust already got you to PCA, this crate is the integration step.

```r
# build the CLI once, then point R at it
# cargo build -p harmorust --release
# export HARMORUST_BIN=$PWD/target/release/harmorust

seurat <- RunPCA(seurat)
seurat <- harmorust::RunHarmony(seurat, group.by.vars = "donor")
seurat <- RunUMAP(seurat, reduction = "harmony")
```

Matrix form (same arguments as `harmony::RunHarmony`):

```r
corrected <- harmorust::RunHarmony(
  pca_matrix, meta_data, vars_use = "donor",
  nclust = 20, return_object = TRUE
)
harmorust::HarmonyConvergencePlot(corrected)
```

See [PARITY.md](PARITY.md) for the Harmony 2.0.5 checklist: API surface, speed tables, and remaining numerical differences.

## GitHub source, Origin mirror

| Role | Repo |
| --- | --- |
| Source of truth | https://github.com/NebilI/harmony-rs (private) |
| Mirror (browse / agents) | https://cursor.com/codebase/nebil-ibrahim/harmony-rs |

GitHub already has `main`. To make the existing Origin repo a GitHub mirror (GitHub stays the source of truth; Origin browse/search stay put), you need the [Cursor GitHub app](https://cursor.com/docs/origin/mirror-github) on `NebilI`, then from WSL / macOS / Linux:

```sh
origin auth login
RECREATE_ORIGIN_MIRROR=1 ./scripts/mirror-github.sh
```

Or in the UI: [cursor.com/codebase](https://cursor.com/codebase) → Sync from GitHub → `NebilI/harmony-rs`. If the native Origin name is taken, delete or detach that copy first.

After the mirror is attached, pushes to the Origin remote go to GitHub; Origin updates after GitHub accepts them. See [Mirror a GitHub repository](https://cursor.com/docs/origin/mirror-github).

Push-only updates to GitHub:

```sh
./scripts/push-github.sh
```

## Run the Rust kernel

Needs a Rust toolchain (`rustc` ≥ 1.89).

```sh
cargo test -p harmorust
cargo run -p harmorust -- demo --cells 400 --batch-strength 2.4
cargo run -p harmorust --release -- bench --cells 4000

# harmorust vs immunogenomics/harmony C++ (needs R + CRAN harmony)
cargo run -p harmorust --release -- compare

# numerical |harmorust − original| (min / max abs diff, scientific notation)
cargo run -p harmorust --release -- parity

# PR gate: fail if a gated max |Δ| exceeds 1e-4
cargo run -p harmorust --release -- parity --check
```

`compare` builds a synthetic atlas at several cell counts, runs both kernels with the same hyperparameters (`max_iter=8`, early stopping off, Harmony's default `nclust`), and prints median wall time plus a speedup column (`original / harmorust`). Repeat it whenever you want fresh numbers:

```sh
cargo run -p harmorust --release -- compare --cells 500,2000,8000 --repeats 7 --warmup 2
cargo run -p harmorust --release -- compare --nclust 5
cargo run -p harmorust --release -- compare --nclust 20
```

On a 4-core Xeon against Harmony 2.0.5, after the Rust `std::arch` `exp`/`ln` (rustc ≥ 1.89), default `nclust` was **1.94×**, `nclust=20` was **2.17×**, and `nclust=5` was **3.12×**. Versus the scalar kernel on this machine that is **1.15× / 1.14× / 1.12×**. The earlier libmvec C landing was a few percent faster (1.99× / 2.23× / 3.26× vs original). Full tables are in [PARITY.md](PARITY.md).

### Numerical difference (`float` vs `f64`)

There is no bit-identical match between Harmony’s C++ `float` loop and harmorust’s `f64`. `parity` runs both kernels once on the same atlas and prints **min / max `|harmorust − original|`** in scientific notation.

```sh
cargo run -p harmorust --release -- parity --cells 200,500,1000 --nclust 1,5
./scripts/parity.sh --nclust 1
```

Compared fields: corrected `embeddings`, cluster-aligned `R` / `Y` / `O`, the outer-round `objective`, and `moe_correct_ridge` (harmorust’s ridge applied to Harmony’s final `R`, vs Harmony `Z_corr`). Same embeddings and hyperparameters as `compare` (`max_iter=8`, early stopping off, `lambda=1`, 20 PCs, 4 batches). Measured against Harmony 2.0.5.

**`nclust = 1` (ridge only)** — no clustering ambiguity. Some embedding entries now match exactly. Max stays around `3e-05` from float GEMM vs our index-walk ridge. `R` and `O` match to `0` or `1.1102e-16`.

| Case | Field | Elements | min \|Δ\| | max \|Δ\| |
| --- | --- | ---: | ---: | ---: |
| atlas-200 | embeddings | 4000 | 0.0000e+00 | 5.2899e-06 |
| atlas-500 | embeddings | 10000 | 0.0000e+00 | 2.9922e-05 |
| atlas-1000 | embeddings | 20000 | 0.0000e+00 | 3.0100e-05 |

**`nclust = 5` (full Harmony loop)** — matching Harmony's k-means++ and R's `runif` cut the old 200-cell O(1) gap (`R` max ≈ 1) to `~3e-02`. 500-cell embeddings stay near `3e-05`. Ridge on Harmony’s `R` is still `≤1e-04`.

| Case | Field | min \|Δ\| | max \|Δ\| |
| --- | --- | ---: | ---: |
| atlas-200 | embeddings | 0.0000e+00 | 3.4487e-02 |
| atlas-200 | moe_correct_ridge | 0.0000e+00 | 9.7752e-06 |
| atlas-500 | embeddings | 0.0000e+00 | 3.4332e-05 |
| atlas-1000 | embeddings | 1.4901e-08 | 1.7810e-04 |

The remaining `3e-02` on the 200-cell / 5-cluster case is Armadillo Lloyd + block shuffle, not the ridge. Full tables, including aligned `R` / `Y` / `O`, are in [PARITY.md](PARITY.md).

### Pull-request gate

PRs fail CI if a gated field’s max `|Δ|` exceeds `1e-4` (`PARITY_MAX_ABS` in `parity.rs`). That rejects O(1) clustering drift while allowing the remaining C++ `float` residual. Unit tests of the gate always run; the live Harmony comparison runs in CI (and locally when CRAN harmony is installed). `HARMORUST_REQUIRE_PARITY=1` makes a missing Harmony install a test failure instead of a skip.

```sh
# JSON in, JSON out (cells × PCs + batch labels)
cargo run -p harmorust -- run < request.json
```

`HarmonyConfig` uses the same names as `harmony::RunHarmony` / `harmony_options`: `theta`, `sigma`, `lambda`, `nclust`, `max_iter`, `block_size`, `tau`, `alpha`.

Match `nclust` to the biology (cell types), not to `batches × types`. Too many clusters lets k-means isolate each donor-type blob; Harmony2 then skips ridge correction for clusters that contain only one batch.

## Run the demo

```sh
cd web
npm install
npm run dev -- --port 43147 --hostname 127.0.0.1
```

The page generates a small atlas, runs the Harmony loop, and plots PC1/PC2 colored by donor or cell type.

## Algorithm (Harmony2)

1. L2-normalise the PCs (cosine space).
2. k-means++ centroids, then soft assignments `R ∝ exp(−dist / σ)`.
3. Online block updates of `R` with the diversity penalty  
   `((2E+1)/(O+E+1))^θ`.
4. Per-cluster ridge: regress the original PCs on `[1 | batch]` weighted by `R`, drop the intercept, subtract the batch coefficients.
5. Repeat until the objective stalls.

Reference: Korsunsky et al., *Nat Methods* (2019) and the Harmony2 preprint. This is an independent Rust implementation of that published method, not a copy of the GPL-3 C++ sources.

## License

MIT. Harmony the paper and the R package are separate works; cite them if you use this in an analysis.
