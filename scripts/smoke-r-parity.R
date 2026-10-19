#!/usr/bin/env Rscript
# Smoke-test the harmorust R API against a tiny synthetic embedding.

args_cmd <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_cmd, value = TRUE)
script <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[[1]]))
} else {
  normalizePath("scripts/smoke-r-parity.R")
}
root <- normalizePath(file.path(dirname(script), ".."))

bin <- file.path(root, "target", "release", "harmorust")
if (!file.exists(bin)) {
  stop("missing release binary at ", bin, " — run cargo build -p harmorust --release")
}
Sys.setenv(HARMORUST_BIN = bin)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}

for (f in list.files(file.path(root, "r", "harmorust", "R"), full.names = TRUE, pattern = "\\.R$")) {
  source(f, local = FALSE)
}

set.seed(1)
n <- 80L
d <- 6L
emb <- matrix(rnorm(n * d), n, d)
meta <- data.frame(
  donor = rep(c("A", "B"), length.out = n),
  stringsAsFactors = FALSE
)

mat <- RunHarmony(
  emb, meta, vars_use = "donor",
  nclust = 4, max_iter = 3, early_stop = FALSE,
  lambda = 1, verbose = FALSE, ncores = 2
)
stopifnot(is.matrix(mat), nrow(mat) == n, ncol(mat) == d)

obj <- RunHarmony(
  emb, meta, vars_use = "donor",
  nclust = 4, max_iter = 3, early_stop = FALSE,
  lambda = 1, verbose = FALSE, return_object = TRUE
)
stopifnot(inherits(obj, "harmony"))
stopifnot(length(obj$kmeans_rounds) >= 1)
stopifnot(length(obj$objective_kmeans) == 1 + sum(obj$kmeans_rounds))

opts <- harmony_options(block.size = 0.1)
stopifnot(inherits(opts, "harmony_options"))

err <- tryCatch(
  RunHarmony(emb, meta, vars_use = "donor", do_pca = TRUE),
  error = function(e) conditionMessage(e)
)
stopifnot(is.character(err), grepl("do_pca", err))

message("smoke-r-parity: ok")
