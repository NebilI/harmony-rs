# Sketch of the seurust → harmorust pipeline.
#
# seurust replaces Seurat C++ kernels (LogNorm, Scale, SNN, …).
# harmorust replaces Harmony's C++ loop. PCA stays in Seurat; only the
# embedding is handed to Rust.

library(Seurat)

# install.packages(
#   c("seurust", "harmorust"),
#   repos = c("https://NebilI.r-universe.dev", "https://cloud.r-project.org")
# )
# library(seurust)
# library(harmorust)

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
