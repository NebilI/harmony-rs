#' Run Harmony with the harmorust kernel
#'
#' Drop-in companion to [harmony::RunHarmony()]. The default method matches
#' Harmony's matrix API (`data_mat`, `meta_data`, `vars_use`). The Seurat
#' method matches `RunHarmony.Seurat` (`group.by.vars`, `reduction.use`,
#' `reduction.save`) so a seurust-preprocessed object can be integrated
#' without changing the rest of the workflow.
#'
#' The numerical work is the Rust crate in this repository
#' (`crates/harmorust`). Wire it through extendr the same way seurust wraps
#' `LogNorm` / `ComputeSNN`. seurust does not implement Harmony itself —
#' it supplies the PCA embedding this kernel consumes.
#'
#' @param object PCA matrix, Seurat object, or SingleCellExperiment.
#' @param group.by.vars,vars_use Covariate name(s) Harmony should remove.
#' @param theta Diversity penalty. Default 2.
#' @param sigma Soft k-means width. Default 0.1.
#' @param lambda Ridge penalty. `NULL` enables Harmony2 automatic lambda.
#' @param nclust Number of soft clusters. Default `min(round(n/30), 100)`.
#' @param reduction.use,reduction Name of the input reduction. Default `"pca"`.
#' @param reduction.save,reduction.name Name of the written reduction.
#' @param dims.use Which PCs to correct.
#' @param project.dim Project Harmony loadings (Seurat). Default TRUE.
#' @param ... Passed to the default method (`harmony_options` fields, `max_iter`).
#' @export
RunHarmony <- function(...) {
  UseMethod("RunHarmony")
}

#' @rdname RunHarmony
#' @export
RunHarmony.default <- function(
    data_mat,
    meta_data,
    vars_use,
    theta = 2,
    sigma = 0.1,
    lambda = NULL,
    nclust = NULL,
    max_iter = 10,
    ...
) {
  if (missing(data_mat) && !missing(meta_data)) {
    ## Allow RunHarmony(matrix, labels) from older sketches.
    data_mat <- meta_data
  }
  if (!is.matrix(data_mat) && !inherits(data_mat, "Matrix")) {
    stop("RunHarmony.default expects a cell embedding matrix")
  }
  labels <- if (is.vector(meta_data) && !is.list(meta_data)) {
    as.character(meta_data)
  } else {
    as.character(meta_data[[vars_use[[1]]]])
  }
  if (length(labels) != nrow(data_mat) && length(labels) != ncol(data_mat)) {
    stop("vars_use / meta_data must have one label per cell")
  }
  if (length(labels) == ncol(data_mat) && length(labels) != nrow(data_mat)) {
    data_mat <- t(as.matrix(data_mat))
  }
  stop(
    "Compile the harmorust extendr staticlib (crates/harmorust) to run the ",
    "Rust kernel from R. Until then use `cargo run -p harmorust -- run` on ",
    "exported PCA embeddings, or the Seurat C++ Harmony package."
  )
}

#' @rdname RunHarmony
#' @export
RunHarmony.Seurat <- function(
    object,
    group.by.vars,
    reduction.use = "pca",
    reduction = NULL,
    dims.use = NULL,
    reduction.save = "harmony",
    reduction.name = NULL,
    project.dim = TRUE,
    theta = 2,
    sigma = 0.1,
    lambda = NULL,
    nclust = NULL,
    ...
) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Install Seurat to use RunHarmony on a Seurat object")
  }
  reduction.use <- reduction %||% reduction.use
  reduction.save <- reduction.name %||% reduction.save
  if (!reduction.use %in% Seurat::Reductions(object)) {
    stop(reduction.use, " cell embeddings not found. Run PCA first (Seurat / seurust).")
  }
  emb <- Seurat::Embeddings(object, reduction = reduction.use)
  if (!is.null(dims.use)) {
    emb <- emb[, dims.use, drop = FALSE]
  }
  meta <- Seurat::FetchData(object, vars = group.by.vars)
  RunHarmony.default(
    data_mat = emb,
    meta_data = meta,
    vars_use = group.by.vars,
    theta = theta,
    sigma = sigma,
    lambda = lambda,
    nclust = nclust,
    ...
  )
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

#' Advanced Harmony options (same names as harmony::harmony_options).
#' @export
harmony_options <- function(
    alpha = 0.2,
    tau = 0,
    block.size = 0.05,
    max.iter.cluster = 4,
    epsilon.cluster = 1e-3,
    epsilon.harmony = 1e-2,
    batch.prop.cutoff = 1e-5
) {
  structure(
    list(
      alpha = alpha,
      tau = tau,
      block.size = block.size,
      max.iter.cluster = max.iter.cluster,
      epsilon.cluster = epsilon.cluster,
      epsilon.harmony = epsilon.harmony,
      batch.prop.cutoff = batch.prop.cutoff
    ),
    class = "harmony_options"
  )
}
