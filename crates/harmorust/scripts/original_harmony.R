#!/usr/bin/env Rscript
# Time immunogenomics/harmony::RunHarmony on the same embeddings the Rust
# compare command generates. Wall time is measured around RunHarmony only.

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
  "embeddings", "batch", "nclust", "max-iter", "warmup", "repeats",
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
warmup <- as.integer(kv$warmup)
repeats <- as.integer(kv$repeats)
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
out_path <- kv$out

if (repeats < 1L) {
  stop("repeats must be >= 1")
}

opts <- harmony_options(
  max.iter.cluster = max_iter_cluster,
  epsilon.cluster = 1e-3,
  epsilon.harmony = if (isTRUE(early_stop)) 1e-2 else -Inf,
  tau = 0,
  block.size = 0.05,
  alpha = 0.2,
  batch.prop.cutoff = 1e-5
)

run_once <- function() {
  set.seed(seed)
  RunHarmony(
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
}

if (warmup > 0L) {
  for (w in seq_len(warmup)) {
    invisible(run_once())
  }
}

now_ms <- function() {
  as.numeric(Sys.time()) * 1000
}

times_ms <- numeric(repeats)
n_iter <- NA_integer_
for (r in seq_len(repeats)) {
  gc(verbose = FALSE)
  t0 <- now_ms()
  obj <- run_once()
  t1 <- now_ms()
  times_ms[[r]] <- t1 - t0
  n_iter <- max(0L, length(obj$objective_harmony) - 1L)
}

fmt <- function(x) sprintf("%.6f", x)
json <- paste0(
  "{",
  "\"version\":\"", as.character(packageVersion("harmony")), "\",",
  "\"n_iter\":", n_iter, ",",
  "\"times_ms\":[", paste(vapply(times_ms, fmt, character(1)), collapse = ","), "]",
  "}\n"
)
con <- file(out_path, open = "wb")
writeChar(json, con, eos = NULL)
close(con)
