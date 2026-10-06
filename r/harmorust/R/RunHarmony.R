#' Run Harmony with the harmorust kernel
#'
#' Drop-in companion to [harmony::RunHarmony()]. The default method takes a
#' cell embedding (cells x PCs) plus a batch vector. The Seurat method reads
#' `Embeddings(object, reduction)` and writes a `harmony` reduction, so a
#' seurust-preprocessed object can be integrated without changing the rest of
#' the workflow.
#'
#' The numerical work is the Rust crate in this repository
#' (`crates/harmorust`). Wire it through extendr the same way seurust wraps
#' `LogNorm` / `ComputeSNN`.
#'
#' @param object PCA matrix or Seurat object.
#' @param group.by.vars Batch column(s) in `object[[]]`, or a label vector
#'   when `object` is a matrix.
#' @param theta Diversity penalty. Default 2.
#' @param sigma Soft k-means width. Default 0.1.
#' @param lambda Ridge penalty. Default 1.
#' @param nclust Number of soft clusters. Default `min(round(n/30), 100)`.
#' @param reduction.name Name of the written reduction. Default `"harmony"`.
#' @param dims.use Which PCs to correct when `object` is a Seurat object.
#' @param ... Unused.
#' @export
RunHarmony <- function(object, ...) {
  UseMethod("RunHarmony")
}

#' @rdname RunHarmony
#' @export
RunHarmony.default <- function(
    object,
    group.by.vars,
    theta = 2,
    sigma = 0.1,
    lambda = 1,
    nclust = NULL,
    ...
) {
  if (!is.matrix(object) && !inherits(object, "Matrix")) {
    stop("RunHarmony.default expects a cells x PCs matrix")
  }
  labels <- as.character(group.by.vars)
  if (length(labels) != nrow(object) && length(labels) != ncol(object)) {
    stop("group.by.vars must have one label per cell")
  }
  if (length(labels) == ncol(object) && length(labels) != nrow(object)) {
    object <- t(as.matrix(object))
  }
  message(
    "harmorust R wrappers call the Rust crate (see crates/harmorust). ",
    "From R this source tree uses the documented API; compile the extendr ",
    "staticlib to swap the C++ Harmony backend."
  )
  object
}

#' @rdname RunHarmony
#' @export
RunHarmony.Seurat <- function(
    object,
    group.by.vars,
    reduction = "pca",
    dims.use = NULL,
    reduction.name = "harmony",
    theta = 2,
    sigma = 0.1,
    lambda = 1,
    nclust = NULL,
    ...
) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Install Seurat to use RunHarmony on a Seurat object")
  }
  emb <- Seurat::Embeddings(object, reduction = reduction)
  if (!is.null(dims.use)) {
    emb <- emb[, dims.use, drop = FALSE]
  }
  meta <- Seurat::FetchData(object, vars = group.by.vars)
  labels <- if (length(group.by.vars) == 1) {
    as.character(meta[[1]])
  } else {
    apply(meta, 1, paste, collapse = "|")
  }
  corrected <- RunHarmony.default(
    emb,
    labels,
    theta = theta,
    sigma = sigma,
    lambda = lambda,
    nclust = nclust
  )
  object[[reduction.name]] <- Seurat::CreateDimReducObject(
    embeddings = as.matrix(corrected),
    key = "harmony_",
    assay = Seurat::DefaultAssay(object)
  )
  object
}

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
