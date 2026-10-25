#!/usr/bin/env Rscript
# Run immunogenomics/harmony::RunHarmony once and dump the result matrices
# so harmorust can compute element-wise |float − f64| differences.

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
lambda <- as.numeric(kv$lambda)
max_iter_cluster <- as.integer(kv[["max-iter-cluster"]])
seed <- as.integer(kv$seed)
out_dir <- kv$out

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

# Harmony 2.0.5 stores Z_corr / Z_orig behind Rcpp getters; older builds
# expose them as fields. R, Y, O, E are public on both.
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

fmt <- function(x) sprintf("%.17g", x)
obj_json <- if (length(objective) == 0L) {
  ""
} else {
  paste(vapply(objective, fmt, character(1)), collapse = ",")
}

json <- paste0(
  "{",
  "\"version\":\"", as.character(packageVersion("harmony")), "\",",
  "\"n_iter\":", n_iter, ",",
  "\"n_cells\":", n_cells, ",",
  "\"n_pcs\":", n_pcs, ",",
  "\"n_clusters\":", n_clusters, ",",
  "\"has_embeddings\":", if (isTRUE(wrote_embeddings)) "true" else "false", ",",
  "\"has_z_orig\":", if (isTRUE(wrote_z_orig)) "true" else "false", ",",
  "\"has_r\":", if (isTRUE(wrote_r)) "true" else "false", ",",
  "\"has_y\":", if (isTRUE(wrote_y)) "true" else "false", ",",
  "\"has_o\":", if (isTRUE(wrote_o)) "true" else "false", ",",
  "\"has_e\":", if (isTRUE(wrote_e)) "true" else "false", ",",
  "\"objective\":[", obj_json, "]",
  "}\n"
)
con <- file(file.path(out_dir, "meta.json"), open = "wb")
writeChar(json, con, eos = NULL)
close(con)
