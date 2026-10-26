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

.write_u32_le <- function(con, x) {
  writeBin(as.integer(x), con, size = 4L, endian = "little", useBytes = TRUE)
}

.write_f32_le <- function(con, x) {
  writeBin(as.double(x), con, size = 4L, endian = "little", useBytes = TRUE)
}

.read_u32_le <- function(con) {
  as.integer(readBin(con, what = "integer", n = 1L, size = 4L, endian = "little"))
}

.read_f32_mat <- function(con) {
  nr <- .read_u32_le(con)
  nc <- .read_u32_le(con)
  if (nr < 1L || nc < 1L) {
    return(matrix(numeric(), nrow = max(0L, nr), ncol = max(0L, nc)))
  }
  vals <- readBin(con, what = "double", n = nr * nc, size = 4L, endian = "little")
  matrix(vals, nrow = nr, ncol = nc, byrow = TRUE)
}

#' Run the Rust Harmony kernel via binary stdin/stdout (`run-bin`).
#'
#' Falls back to the JSON `run` path if the binary protocol is unavailable.
#'
#' @param embeddings Numeric matrix, cells × PCs.
#' @param covariates List of character vectors (one per `vars_use`).
#' @param theta Numeric vector, length 1 or `length(covariates)`.
#' @param lambda `NULL` for automatic lambda, else numeric length 1 or per covariate.
#' @param config Named list matching `HarmonyConfig` JSON fields.
#' @param ncores Sets `RAYON_NUM_THREADS` for the child process.
#' @param verbose Print a short progress message.
#' @return Parsed list from the CLI response.
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
  if (is.null(lambda)) {
    config$lambda <- NULL
  } else if (length(lambda) == 1L) {
    config$lambda <- as.numeric(lambda[[1]])
  }

  header <- list(
    n_cells = nrow(embeddings),
    n_pcs = ncol(embeddings),
    covariates = covariates,
    theta = as.numeric(theta),
    config = config
  )
  if (!is.null(lambda)) {
    header$lambda <- as.numeric(lambda)
  }

  tmp_in <- tempfile(fileext = ".bin")
  tmp_out <- tempfile(fileext = ".bin")
  tmp_err <- tempfile(fileext = ".log")
  on.exit(unlink(c(tmp_in, tmp_out, tmp_err)), add = TRUE)

  header_json <- jsonlite::toJSON(header, auto_unbox = TRUE, null = "null", digits = NA)
  header_raw <- charToRaw(header_json)
  con <- file(tmp_in, open = "wb")
  writeBin(charToRaw("HRu1"), con, useBytes = TRUE)
  .write_u32_le(con, length(header_raw))
  writeBin(header_raw, con, useBytes = TRUE)
  .write_f32_le(con, as.numeric(t(embeddings)))
  close(con)

  if (isTRUE(verbose)) {
    message("Running harmorust kernel (", bin, " run-bin)")
  }

  status <- system2(
    bin,
    args = "run-bin",
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
    return(.call_harmorust_kernel_json(
      bin, embeddings, covariates, theta, lambda, config, ncores, verbose, err
    ))
  }

  con <- file(tmp_out, open = "rb")
  on.exit(close(con), add = TRUE)
  magic <- rawToChar(readBin(con, what = "raw", n = 4L))
  if (!identical(magic, "HRu1")) {
    stop("harmorust run-bin: bad magic", call. = FALSE)
  }
  meta_len <- .read_u32_le(con)
  meta_raw <- readBin(con, what = "raw", n = meta_len)
  meta <- jsonlite::fromJSON(rawToChar(meta_raw), simplifyVector = TRUE)
  embeddings_out <- .read_f32_mat(con)
  r <- .read_f32_mat(con)
  y <- .read_f32_mat(con)
  o <- .read_f32_mat(con)
  e <- .read_f32_mat(con)
  list(
    embeddings = embeddings_out,
    r = r,
    y = y,
    o = o,
    e = e,
    n_iter = meta$n_iter,
    converged = isTRUE(meta$converged),
    n_clusters = meta$n_clusters,
    objective = as.numeric(meta$objective),
    objective_kmeans = as.numeric(meta$objective_kmeans),
    kmeans_rounds = as.integer(meta$kmeans_rounds),
    rng_state = if (!is.null(meta$rng_state)) as.integer(meta$rng_state) else NULL
  )
}

.call_harmorust_kernel_json <- function(
    bin, embeddings, covariates, theta, lambda, config, ncores, verbose, prev_err
) {
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
    message("Falling back to harmorust JSON run")
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
    prev_err
  }
  if (!identical(as.integer(status), 0L)) {
    stop(
      "harmorust run failed",
      if (nzchar(err)) paste0(":\n", err) else paste0(" (exit ", status, ")"),
      call. = FALSE
    )
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
      E = if (!is.null(result$e)) as.matrix(result$e) else NULL,
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
