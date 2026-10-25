#' Locate the harmorust CLI binary.
#'
#' Resolution order: `HARMORUST_BIN`, `Sys.which("harmorust")`, then common
#' cargo `target/release` paths relative to this source tree / working directory.
#'
#' @return Absolute path to the binary.
#' @keywords internal
.find_harmorust_bin <- function() {
  env <- Sys.getenv("HARMORUST_BIN", unset = "")
  if (nzchar(env) && file.exists(env)) {
    return(normalizePath(env, winslash = "/", mustWork = TRUE))
  }
  on_path <- Sys.which("harmorust")
  if (nzchar(on_path)) {
    return(normalizePath(on_path, winslash = "/", mustWork = TRUE))
  }
  candidates <- c(
    file.path(getwd(), "target", "release", "harmorust"),
    file.path(getwd(), "..", "target", "release", "harmorust"),
    file.path(getwd(), "..", "..", "target", "release", "harmorust")
  )
  pkg_dir <- tryCatch(system.file(package = "harmorust"), error = function(e) "")
  if (nzchar(pkg_dir)) {
    candidates <- c(
      candidates,
      file.path(pkg_dir, "bin", "harmorust"),
      file.path(dirname(dirname(pkg_dir)), "target", "release", "harmorust")
    )
  }
  for (path in candidates) {
    if (file.exists(path)) {
      return(normalizePath(path, winslash = "/", mustWork = TRUE))
    }
  }
  stop(
    "Could not find the harmorust binary. Build it with ",
    "`cargo build -p harmorust --release`, put it on PATH, or set HARMORUST_BIN.",
    call. = FALSE
  )
}

#' Run the Rust Harmony kernel via JSON stdin/stdout.
#'
#' @param embeddings Numeric matrix, cells × PCs.
#' @param covariates List of character vectors (one per `vars_use`).
#' @param theta Numeric vector, length 1 or `length(covariates)`.
#' @param lambda `NULL` for automatic lambda, else numeric length 1 or per covariate.
#' @param config Named list matching `HarmonyConfig` JSON fields.
#' @param ncores Sets `RAYON_NUM_THREADS` for the child process.
#' @param verbose Print a short progress message.
#' @return Parsed list from the CLI JSON response.
#' @keywords internal
.call_harmorust_kernel <- function(
    embeddings,
    covariates,
    theta,
    lambda,
    config,
    ncores = 1L,
    verbose = TRUE
) {
  bin <- .find_harmorust_bin()
  emb_rows <- lapply(seq_len(nrow(embeddings)), function(i) {
    as.numeric(embeddings[i, , drop = TRUE])
  })
  req <- list(
    embeddings = emb_rows,
    covariates = covariates,
    theta = as.numeric(theta),
    config = config
  )
  if (is.null(lambda)) {
    req$config$lambda <- NULL
  } else {
    req$lambda <- as.numeric(lambda)
    if (length(lambda) == 1L) {
      req$config$lambda <- as.numeric(lambda[[1]])
    }
  }

  tmp_in <- tempfile(fileext = ".json")
  tmp_out <- tempfile(fileext = ".json")
  tmp_err <- tempfile(fileext = ".log")
  on.exit(unlink(c(tmp_in, tmp_out, tmp_err)), add = TRUE)

  jsonlite::write_json(req, tmp_in, auto_unbox = TRUE, null = "null", digits = NA)
  if (isTRUE(verbose)) {
    message("Running harmorust kernel (", bin, ")")
  }

  status <- system2(
    bin,
    args = "run",
    stdin = tmp_in,
    stdout = tmp_out,
    stderr = tmp_err,
    env = sprintf("RAYON_NUM_THREADS=%d", max(1L, as.integer(ncores)))
  )
  err <- if (file.exists(tmp_err)) {
    paste(readLines(tmp_err, warn = FALSE), collapse = "\n")
  } else {
    ""
  }
  if (!identical(as.integer(status), 0L)) {
    stop(
      "harmorust run failed",
      if (nzchar(err)) paste0(":\n", err) else paste0(" (exit ", status, ")"),
      call. = FALSE
    )
  }
  if (!file.exists(tmp_out) || !length(readLines(tmp_out, n = 1, warn = FALSE))) {
    stop("harmorust produced no JSON output", if (nzchar(err)) paste0(":\n", err) else "", call. = FALSE)
  }
  jsonlite::fromJSON(tmp_out, simplifyVector = TRUE)
}

#' Build a Harmony-like return object from a kernel response.
#' @keywords internal
.as_harmony_object <- function(result, row_names = NULL, col_names = NULL) {
  emb <- as.matrix(result$embeddings)
  if (!is.null(row_names) && length(row_names) == nrow(emb)) {
    rownames(emb) <- row_names
  }
  if (!is.null(col_names) && length(col_names) == ncol(emb)) {
    colnames(emb) <- col_names
  }
  structure(
    list(
      embeddings = emb,
      Z_corr = t(emb),
      R = as.matrix(result$r),
      Y = as.matrix(result$y),
      O = as.matrix(result$o),
      objective_harmony = as.numeric(result$objective),
      objective_kmeans = as.numeric(result$objective_kmeans),
      kmeans_rounds = as.integer(result$kmeans_rounds),
      n_iter = as.integer(result$n_iter),
      converged = isTRUE(result$converged),
      n_clusters = as.integer(result$n_clusters)
    ),
    class = c("harmony", "harmorust_harmony")
  )
}
