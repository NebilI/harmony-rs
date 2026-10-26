#!/usr/bin/env Rscript
# Run immunogenomics/harmony::RunHarmony once and dump the result matrices
# so harmorust can compute element-wise |float − f64| differences.
#
# Also records the compiled scalar type, BLAS/LAPACK, RcppArmadillo version,
# and optional stage dumps (setup / init / each cluster+correct round).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) %% 2L != 0L) {
  stop("expected --flag value pairs")
}
kv <- list()
i <- 1L
while (i <= length(args)) {
  key <- sub("^--", "", args[[i]])
  kv[[key]] <- args[[i + 1L]]
  i <- i + 2L
}
need <- c(
  "embeddings", "batch", "nclust", "max-iter",
  "ncores", "early-stop", "theta", "sigma", "lambda", "max-iter-cluster",
  "seed", "out"
)
missing <- need[!need %in% names(kv)]
if (length(missing) > 0L) {
  stop(paste("missing flags:", paste(missing, collapse = ", ")))
}

user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(harmony)
})

emb <- as.matrix(read.csv(kv$embeddings, check.names = FALSE))
storage.mode(emb) <- "double"
meta <- read.csv(kv$batch, stringsAsFactors = FALSE)
if (!"batch" %in% names(meta)) {
  stop("batch CSV must have a 'batch' column")
}

nclust <- as.integer(kv$nclust)
max_iter <- as.integer(kv[["max-iter"]])
ncores <- as.integer(kv$ncores)
early_stop <- as.logical(kv[["early-stop"]])
theta <- as.numeric(kv$theta)
sigma <- as.numeric(kv$sigma)
lambda <- if (identical(kv$lambda, "auto") || identical(kv$lambda, "NULL")) {
  NULL
} else {
  as.numeric(kv$lambda)
}
max_iter_cluster <- as.integer(kv[["max-iter-cluster"]])
seed <- as.integer(kv$seed)
out_dir <- kv$out
dump_stages <- !identical(kv$stages, "FALSE")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

opts <- harmony_options(
  max.iter.cluster = max_iter_cluster,
  epsilon.cluster = 1e-3,
  epsilon.harmony = if (isTRUE(early_stop)) 1e-2 else -Inf,
  tau = 0,
  block.size = 0.05,
  alpha = 0.2,
  batch.prop.cutoff = 1e-5
)

set.seed(seed)
obj <- RunHarmony(
  emb,
  meta$batch,
  theta = theta,
  sigma = sigma,
  lambda = lambda,
  nclust = nclust,
  max_iter = max_iter,
  early_stop = early_stop,
  ncores = ncores,
  verbose = FALSE,
  return_object = TRUE,
  .options = opts
)

as_mat <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  m <- tryCatch(as.matrix(x), error = function(e) NULL)
  if (is.null(m)) {
    return(NULL)
  }
  storage.mode(m) <- "double"
  m
}

try_get <- function(expr) {
  tryCatch(as_mat(expr), error = function(e) NULL)
}

write_mat <- function(path, m) {
  if (is.null(m)) {
    return(invisible(FALSE))
  }
  con <- file(path, open = "wt")
  on.exit(close(con), add = TRUE)
  nr <- nrow(m)
  nc <- ncol(m)
  if (is.null(nr) || is.null(nc) || nr < 1L || nc < 1L) {
    return(invisible(FALSE))
  }
  for (r in seq_len(nr)) {
    writeLines(paste(sprintf("%.17g", m[r, ]), collapse = ","), con)
  }
  invisible(TRUE)
}

orient_cells_by_pc <- function(m, n_cells, n_pcs) {
  if (is.null(m)) {
    return(NULL)
  }
  if (nrow(m) == n_cells && ncol(m) == n_pcs) {
    return(m)
  }
  if (nrow(m) == n_pcs && ncol(m) == n_cells) {
    return(t(m))
  }
  m
}

n_cells <- nrow(emb)
n_pcs <- ncol(emb)

z_corr <- try_get(obj$getZcorr())
if (is.null(z_corr)) {
  z_corr <- try_get(obj$Z_corr)
}
embeddings <- orient_cells_by_pc(z_corr, n_cells, n_pcs)
if (is.null(embeddings)) {
  stop("Harmony return_object has no Z_corr / embedding matrix")
}

z_orig <- try_get(obj$getZorig())
if (is.null(z_orig)) {
  z_orig <- try_get(obj$Z_orig)
}
r <- try_get(obj$R)
y <- try_get(obj$Y)
o <- try_get(obj$O)
e <- try_get(obj$E)
objective <- tryCatch(as.numeric(obj$objective_harmony), error = function(e) numeric())
if (length(objective) == 0L || all(is.na(objective))) {
  objective <- numeric()
}
kmeans_rounds <- tryCatch(as.integer(obj$kmeans_rounds), error = function(e) integer())
n_iter <- max(0L, length(objective) - 1L)
n_clusters <- if (!is.null(r)) nrow(r) else nclust

wrote_embeddings <- write_mat(file.path(out_dir, "embeddings.csv"), embeddings)
wrote_z_orig <- write_mat(
  file.path(out_dir, "z_orig.csv"),
  if (!is.null(z_orig)) z_orig else t(emb)
)
wrote_r <- write_mat(file.path(out_dir, "R.csv"), r)
wrote_y <- write_mat(file.path(out_dir, "Y.csv"), y)
wrote_o <- write_mat(file.path(out_dir, "O.csv"), o)
wrote_e <- write_mat(file.path(out_dir, "E.csv"), e)

# Scalar type: if every getZorig() value is exactly float-representable,
# Harmony compiled with SCALAR=float (the input CSV has full f64).
z_probe <- if (!is.null(z_orig)) as.numeric(z_orig) else as.numeric(embeddings)
scalar_type <- if (length(z_probe) > 0L && all(z_probe == as.numeric(as.single(z_probe)))) {
  "float"
} else {
  "double"
}

blas <- tryCatch(as.character(extSoftVersion()[["BLAS"]]), error = function(e) "")
lapack <- tryCatch(as.character(La_library()), error = function(e) "")
rcpparma <- tryCatch(
  as.character(utils::packageVersion("RcppArmadillo")),
  error = function(e) ""
)

json_escape <- function(s) {
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub("\"", "\\\"", s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  s
}

fmt <- function(x) sprintf("%.17g", x)
obj_json <- if (length(objective) == 0L) {
  ""
} else {
  paste(vapply(objective, fmt, character(1)), collapse = ",")
}
kr_json <- if (length(kmeans_rounds) == 0L) {
  ""
} else {
  paste(as.integer(kmeans_rounds), collapse = ",")
}

# Optional stage dumps: drive the Rcpp module the same way RunHarmony does.
n_stages <- 0L
if (isTRUE(dump_stages)) {
  stages_dir <- file.path(out_dir, "stages")
  dir.create(stages_dir, recursive = TRUE, showWarnings = FALSE)
  stage_ok <- tryCatch(
    {
      set.seed(seed)
      data_mat <- t(emb)
      phi <- Matrix::t(Matrix::sparse.model.matrix(~0 + as.factor(meta$batch)))
      B_vec <- nlevels(as.factor(meta$batch))
      N_b <- Matrix::rowSums(phi)
      sigma_vec <- if (nclust > 1L) rep(sigma, nclust) else sigma
      theta_vec <- rep(theta, B_vec)
      tau <- 0
      theta_vec <- theta_vec * (1 - exp(-(N_b / (nclust * tau))^2))
      lambda_vec <- if (is.null(lambda)) {
        -1
      } else {
        c(0, rep(lambda, B_vec))
      }
      ho <- methods::new(harmony)
      ho$setup(
        data_mat, phi, sigma_vec, theta_vec, lambda_vec, 0.2,
        max_iter_cluster, 1e-3,
        if (isTRUE(early_stop)) 1e-2 else -Inf,
        nclust, 0.05, B_vec, 1e-5, FALSE
      )
      write_mat(file.path(stages_dir, "00_setup_Zcorr.csv"), try_get(ho$getZcorr()))
      ho$init_cluster_cpp()
      write_mat(file.path(stages_dir, "01_init_Y.csv"), try_get(ho$Y))
      write_mat(file.path(stages_dir, "01_init_R.csv"), try_get(ho$R))
      write_mat(file.path(stages_dir, "01_init_O.csv"), try_get(ho$O))
      write_mat(file.path(stages_dir, "01_init_E.csv"), try_get(ho$E))
      n_st <- 1L
      for (it in seq_len(max_iter)) {
        ho$cluster_cpp()
        write_mat(
          file.path(stages_dir, sprintf("%02d_cluster_R.csv", it + 1L)),
          try_get(ho$R)
        )
        write_mat(
          file.path(stages_dir, sprintf("%02d_cluster_Y.csv", it + 1L)),
          try_get(ho$Y)
        )
        ho$moe_correct_ridge_cpp()
        write_mat(
          file.path(stages_dir, sprintf("%02d_correct_Z.csv", it + 1L)),
          try_get(ho$getZcorr())
        )
        n_st <- n_st + 1L
        if (isTRUE(ho$check_convergence(1L))) {
          break
        }
      }
      n_st
    },
    error = function(e) {
      writeLines(conditionMessage(e), file.path(out_dir, "stages_error.txt"))
      0L
    }
  )
  n_stages <- as.integer(stage_ok)
}

json <- paste0(
  "{",
  "\"version\":\"", json_escape(as.character(packageVersion("harmony"))), "\",",
  "\"n_iter\":", n_iter, ",",
  "\"n_cells\":", n_cells, ",",
  "\"n_pcs\":", n_pcs, ",",
  "\"n_clusters\":", n_clusters, ",",
  "\"scalar_type\":\"", json_escape(scalar_type), "\",",
  "\"blas\":\"", json_escape(blas), "\",",
  "\"lapack\":\"", json_escape(lapack), "\",",
  "\"rcpparmadillo\":\"", json_escape(rcpparma), "\",",
  "\"ncores\":", ncores, ",",
  "\"n_stages\":", n_stages, ",",
  "\"has_embeddings\":", if (isTRUE(wrote_embeddings)) "true" else "false", ",",
  "\"has_z_orig\":", if (isTRUE(wrote_z_orig)) "true" else "false", ",",
  "\"has_r\":", if (isTRUE(wrote_r)) "true" else "false", ",",
  "\"has_y\":", if (isTRUE(wrote_y)) "true" else "false", ",",
  "\"has_o\":", if (isTRUE(wrote_o)) "true" else "false", ",",
  "\"has_e\":", if (isTRUE(wrote_e)) "true" else "false", ",",
  "\"kmeans_rounds\":[", kr_json, "],",
  "\"objective\":[", obj_json, "]",
  "}\n"
)
con <- file(file.path(out_dir, "meta.json"), open = "wb")
writeChar(json, con, eos = NULL)
close(con)
