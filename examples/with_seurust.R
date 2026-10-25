# Sketch of the seurust → harmorust pipeline.
#
# seurust replaces Seurat C++ kernels (LogNorm, Scale, SNN, …).
# harmorust replaces Harmony's C++ loop. PCA stays in Seurat; only the
# embedding is handed to Rust.

library(Seurat)

# library(seurust)    # install.packages("seurust", repos = "https://nebili.r-universe.dev")
# library(harmorust)  # needs HARMORUST_BIN or harmorust on PATH
# Sys.setenv(HARMORUST_BIN = "target/release/harmorust")

# counts <- Read10X("path/to/filtered_feature_bc_matrix")
# obj <- CreateSeuratObject(counts)
# obj$donor <- donor_id

# seurust-backed preprocessing
# obj <- NormalizeData(obj)
# obj <- FindVariableFeatures(obj)
# obj <- ScaleData(obj)
# obj <- RunPCA(obj)

# Harmony on those PCs — same call you already use
# obj <- harmorust::RunHarmony(obj, group.by.vars = "donor")
# obj <- FindNeighbors(obj, reduction = "harmony")
# obj <- FindClusters(obj)
# obj <- RunUMAP(obj, reduction = "harmony")

message("See r/harmorust and crates/harmorust. Run the Rust kernel with:")
message("  cargo run -p harmorust -- demo --cells 400")
