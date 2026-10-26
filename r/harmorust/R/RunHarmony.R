#' Run Harmony with the harmorust kernel
#'
#' Drop-in companion to [harmony::RunHarmony()]. The default method matches
#' Harmony's matrix API (`data_mat`, `meta_data`, `vars_use`). Seurat and
#' SingleCellExperiment methods match the original package so a
#' seurust-preprocessed object can be integrated without changing the rest of
#' the workflow.
#'
#' The numerical work is the Rust crate in this repository
#' (`crates/harmorust`), invoked through the `harmorust` CLI (`HARMORUST_BIN`
#' or `PATH`).
#'
#' @param object PCA matrix, Seurat object, or SingleCellExperiment.
#' @param data_mat Cell embedding matrix (cells × PCs or PCs × cells).
#' @param meta_data Data frame of covariates, or a batch vector.
#' @param vars_use,group.by.vars Covariate name(s) Harmony should remove.
#' @param theta Diversity penalty. Default 2 per covariate.
#' @param sigma Soft k-means width. Default 0.1.
#' @param lambda Ridge penalty. `NULL` enables Harmony2 automatic lambda.
#' @param nclust Number of soft clusters. Default `min(round(n/30), 100)`.
#' @param max_iter Outer Harmony rounds. Default 10.
#' @param early_stop Stop when the objective stalls. Default TRUE.
#' @param ncores Rayon threads for the Rust kernel (`RAYON_NUM_THREADS`).
#' @param plot_convergence Draw [HarmonyConvergencePlot()] when TRUE.
#' @param return_object Return a Harmony-like object instead of the matrix.
#' @param verbose Print progress messages.
#' @param .options Advanced options from [harmony_options()].
#' @param reduction.use,reduction Name of the input reduction. Default `"pca"`.
#' @param reduction.save,reduction.name Name of the written reduction.
#' @param dims.use Which PCs to correct.
#' @param project.dim Project Harmony loadings (Seurat). Default TRUE.
#' @param ... Must be empty (legacy Harmony args are rejected).
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
    theta = NULL,
    sigma = 0.1,
    lambda = NULL,
    nclust = NULL,
    max_iter = 10,
    early_stop = TRUE,
    ncores = 1,
    plot_convergence = FALSE,
    return_object = FALSE,
    verbose = TRUE,
    .options = harmony_options(),
    ...
) {
  .check_legacy_args(...)
  if (!inherits(.options, "harmony_options")) {
    stop("Error: .options must be created from harmony_options()!")
  }

  if (!(is.data.frame(meta_data) || inherits(meta_data, "DataFrame"))) {
    if (length(meta_data) %in% dim(data_mat)) {
      meta_data <- data.frame(batch_variable = meta_data, stringsAsFactors = FALSE)
      vars_use <- "batch_variable"
    } else {
      stop(
        "meta_data must be either a data.frame or a vector with batch values for each cell"
      )
    }
  }
  meta_data <- as.data.frame(meta_data, stringsAsFactors = FALSE)

  if (is.null(vars_use) || any(!vars_use %in% colnames(meta_data))) {
    stop(gettextf("must provide variables names (e.g. vars_use=%s)", sQuote("stim")))
  }

  N <- nrow(meta_data)
  data_mat <- as.matrix(data_mat)
  # Harmony-R accepts cells×PCs or PCs×cells; the Rust kernel wants cells×PCs.
  if (ncol(data_mat) == N && nrow(data_mat) != N) {
    if (verbose) {
      message("Transposing data matrix")
    }
    data_mat <- t(data_mat)
  } else if (nrow(data_mat) != N) {
    stop("number of labels do not correspond to number of samples in data matrix")
  }

  if (is.null(nclust)) {
    nclust <- min(round(N / 30), 100)
  }
  if (is.null(theta)) {
    theta <- rep(2, length(vars_use))
  } else if (length(theta) != length(vars_use)) {
    stop("Please specify theta for each variable")
  }

  if (!is.null(lambda)) {
    if (!all(lambda > 0)) {
      stop("Provided lambdas must be positive")
    }
    if (length(lambda) != 1L && length(lambda) != length(vars_use)) {
      stop(paste0(
        "You specified a lambda value for each ",
        "covariate but the number of lambdas specified (",
        length(lambda), ") and the number of covariates (",
        length(vars_use), ") mismatch."
      ))
    }
  }

  covariates <- lapply(vars_use, function(v) as.character(meta_data[[v]]))

  config <- list(
    theta = theta[[1]],
    sigma = sigma[[1]],
    lambda = if (is.null(lambda)) NULL else lambda[[1]],
    nclust = as.integer(nclust),
    max_iter = as.integer(max_iter),
    max_iter_cluster = as.integer(.options$max.iter.cluster),
    epsilon_cluster = .options$epsilon.cluster,
    epsilon_harmony = .options$epsilon.harmony,
    block_size = .options$block.size,
    alpha = .options$alpha,
    tau = .options$tau,
    batch_prop_cutoff = .options$batch.prop.cutoff,
    seed = 1L,
    early_stop = isTRUE(early_stop)
  )
  if (!exists(".Random.seed", envir = .GlobalEnv)) {
    set.seed(NULL)
  }
  config$rng_state <- as.integer(get(".Random.seed", envir = .GlobalEnv))

  if (verbose && is.null(lambda)) {
    message("Using automatic lambda estimation")
  }

  result <- .call_harmorust_kernel(
    embeddings = data_mat,
    covariates = covariates,
    theta = theta,
    lambda = lambda,
    config = config,
    ncores = ncores,
    verbose = verbose
  )

  if (!is.null(result$rng_state)) {
    assign(".Random.seed", as.integer(result$rng_state), envir = .GlobalEnv)
  }

  obj <- .as_harmony_object(
    result,
    row_names = rownames(data_mat),
    col_names = colnames(data_mat)
  )

  if (isTRUE(plot_convergence)) {
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      print(HarmonyConvergencePlot(obj))
    } else {
      warning("plot_convergence=TRUE requires ggplot2; skipping plot")
    }
  }

  if (isTRUE(return_object)) {
    return(obj)
  }
  obj$embeddings
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
    ...
) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Running Harmony on a Seurat object requires Seurat")
  }
  reduction.use <- if (!is.null(reduction)) reduction else reduction.use
  reduction.save <- if (!is.null(reduction.name)) reduction.name else reduction.save
  if (!reduction.use %in% Seurat::Reductions(object = object)) {
    stop(paste(
      reduction.use, "cell embeddings not found in Seurat object.",
      "For a Seurat preprocessing walkthrough, please refer to the vignette"
    ))
  }
  embedding <- Seurat::Embeddings(object, reduction = reduction.use)
  if (is.null(dims.use)) {
    dims.use <- seq_len(ncol(embedding))
  }
  dims_avail <- seq_len(ncol(embedding))
  if (!all(dims.use %in% dims_avail)) {
    stop(paste(
      "trying to use more dimensions than computed. Rerun dimension reduction",
      "with more dimensions or run Harmony with fewer dimensions"
    ))
  }
  if (length(dims.use) == 1) {
    stop("only specified one dimension in dims.use")
  }
  metavars_df <- Seurat::FetchData(
    object, group.by.vars,
    cells = Seurat::Cells(x = object[[reduction.use]])
  )
  harmonyEmbed <- RunHarmony(
    data_mat = embedding[, dims.use, drop = FALSE],
    meta_data = metavars_df,
    vars_use = group.by.vars,
    return_object = FALSE,
    ...
  )
  reduction.key <- Seurat::Key(reduction.save, quiet = TRUE)
  rownames(harmonyEmbed) <- rownames(embedding)
  colnames(harmonyEmbed) <- paste0(reduction.key, seq_len(ncol(harmonyEmbed)))
  object[[reduction.save]] <- Seurat::CreateDimReducObject(
    embeddings = harmonyEmbed,
    stdev = as.numeric(apply(harmonyEmbed, 2, stats::sd)),
    assay = Seurat::DefaultAssay(object = object[[reduction.use]]),
    key = reduction.key
  )
  if (isTRUE(project.dim)) {
    object <- Seurat::ProjectDim(
      object,
      reduction = reduction.save,
      overwrite = TRUE,
      verbose = FALSE
    )
  }
  object
}

#' @rdname RunHarmony
#' @export
RunHarmony.SingleCellExperiment <- function(
    object,
    group.by.vars,
    dims.use = NULL,
    verbose = TRUE,
    reduction.save = "HARMONY",
    ...
) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("Running Harmony on a SingleCellExperiment requires SingleCellExperiment")
  }
  if (!"PCA" %in% SingleCellExperiment::reducedDimNames(object)) {
    stop("PCA must be computed before running Harmony.")
  }
  pca_embedding <- SingleCellExperiment::reducedDim(object, "PCA")
  if (is.null(dims.use)) {
    dims.use <- seq_len(ncol(pca_embedding))
  }
  dims_avail <- seq_len(ncol(pca_embedding))
  if (!all(dims.use %in% dims_avail)) {
    stop(paste(
      "trying to use more dimensions than computed with PCA. Rerun",
      "PCA with more dimensions or use fewer PCs"
    ))
  }
  metavars_df <- as.data.frame(SingleCellExperiment::colData(object))
  if (!all(group.by.vars %in% colnames(metavars_df))) {
    stop("Trying to integrate over variables missing in colData")
  }
  harmonyEmbed <- RunHarmony(
    data_mat = pca_embedding[, dims.use, drop = FALSE],
    meta_data = metavars_df,
    vars_use = group.by.vars,
    return_object = FALSE,
    verbose = verbose,
    ...
  )
  rownames(harmonyEmbed) <- rownames(metavars_df)
  colnames(harmonyEmbed) <- paste0(reduction.save, "_", seq_len(ncol(harmonyEmbed)))
  SingleCellExperiment::reducedDim(object, reduction.save) <- harmonyEmbed
  object
}
