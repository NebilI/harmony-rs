# harmorust

Rust kernels for [Harmony](https://github.com/immunogenomics/harmony), with the
same `RunHarmony()` API as the immunogenomics R package. Pair it with
[seurust](https://github.com/NebilI/seurust): Seurat / seurust produce the PCA
embedding, harmorust corrects it.

[![r-universe version](https://NebilI.r-universe.dev/harmorust/badges/version)](https://NebilI.r-universe.dev/harmorust)
[![r-universe status](https://NebilI.r-universe.dev/harmorust/badges/checks)](https://NebilI.r-universe.dev/harmorust)

## Install

Requires R ≥ 4.2 and, for source installs, a Rust toolchain (`rustc` + Cargo ≥ 1.89).

### From r-universe

```r
install.packages(
  "harmorust",
  repos = c("https://NebilI.r-universe.dev", "https://cloud.r-project.org")
)
```

### From GitHub

```r
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("NebilI/harmony-rs", subdir = "r/harmorust")
```

### From a release tarball

```r
install.packages("harmorust_0.1.0.tar.gz", repos = NULL, type = "source")
```

Source installs compile the `harmorust` CLI and put it in the package `bin`
directory. Override with `HARMORUST_BIN` only if you want a different binary.

## Example

```r
library(harmorust)

# matrix form (cells × PCs) — same arguments as harmony::RunHarmony
corrected <- RunHarmony(
  pca_matrix, meta_data, vars_use = "donor",
  nclust = 20, return_object = TRUE
)
HarmonyConvergencePlot(corrected)

# Seurat / seurust workflow
# seurat <- RunPCA(seurat)
# seurat <- harmorust::RunHarmony(seurat, group.by.vars = "donor")
```

See the [repository README](https://github.com/NebilI/harmony-rs) for the Rust
crate, parity notes, and publishing pipeline.
