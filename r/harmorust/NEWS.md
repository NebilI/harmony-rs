# harmorust 0.1.0

First public release of the R package (version 0.1).

* `RunHarmony()` for a PCA matrix, a Seurat object, or a SingleCellExperiment.
* `harmony_options()` and `HarmonyConvergencePlot()` match the immunogenomics
  Harmony public surface.
* Source installs compile the Rust CLI with Cargo and place it in the package
  `bin` directory so `install.packages("harmorust")` from
  [r-universe](https://NebilI.r-universe.dev) works without a separate
  `HARMORUST_BIN`.
