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
| `r/harmorust` | R package skeleton with the `RunHarmony()` signature, ready for an extendr bind like seurust. |
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
seurat <- RunPCA(seurat)
seurat <- harmorust::RunHarmony(seurat, group.by.vars = "donor")
seurat <- RunUMAP(seurat, reduction = "harmony")
```

## Run the Rust kernel

Needs a Rust toolchain (`rustc` ≥ 1.81).

```sh
cargo test -p harmorust
cargo run -p harmorust -- demo --cells 400 --batch-strength 2.4

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
