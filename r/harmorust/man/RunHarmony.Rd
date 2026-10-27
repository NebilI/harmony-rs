\name{RunHarmony}
\alias{RunHarmony}
\alias{RunHarmony.default}
\alias{RunHarmony.Seurat}
\alias{RunHarmony.SingleCellExperiment}
\title{Run Harmony with the harmorust kernel}
\description{
  Drop-in companion to \code{harmony::RunHarmony()}. The default method
  matches Harmony's matrix API (\code{data_mat}, \code{meta_data},
  \code{vars_use}). Seurat and SingleCellExperiment methods match the
  original package so a seurust-preprocessed object can be integrated
  without changing the rest of the workflow.
}
\usage{
RunHarmony(...)

\method{RunHarmony}{default}(
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
)

\method{RunHarmony}{Seurat}(
  object,
  group.by.vars,
  reduction.use = "pca",
  reduction = NULL,
  dims.use = NULL,
  reduction.save = "harmony",
  reduction.name = NULL,
  project.dim = TRUE,
  ...
)

\method{RunHarmony}{SingleCellExperiment}(
  object,
  group.by.vars,
  dims.use = NULL,
  verbose = TRUE,
  reduction.save = "HARMONY",
  ...
)
}
\arguments{
  \item{object}{PCA matrix, Seurat object, or SingleCellExperiment.}
  \item{data_mat}{Cell embedding matrix (cells \eqn{\times} PCs or PCs
    \eqn{\times} cells).}
  \item{meta_data}{Data frame of covariates, or a batch vector.}
  \item{vars_use, group.by.vars}{Covariate name(s) Harmony should remove.}
  \item{theta}{Diversity penalty. Default 2 per covariate.}
  \item{sigma}{Soft k-means width. Default 0.1.}
  \item{lambda}{Ridge penalty. \code{NULL} enables Harmony2 automatic lambda.}
  \item{nclust}{Number of soft clusters. Default
    \code{min(round(n/30), 100)}.}
  \item{max_iter}{Outer Harmony rounds. Default 10.}
  \item{early_stop}{Stop when the objective stalls. Default \code{TRUE}.}
  \item{ncores}{Rayon threads for the Rust kernel
    (\code{RAYON_NUM_THREADS}).}
  \item{plot_convergence}{Draw \code{\link{HarmonyConvergencePlot}} when
    \code{TRUE}.}
  \item{return_object}{Return a Harmony-like object instead of the matrix.}
  \item{verbose}{Print progress messages.}
  \item{.options}{Advanced options from \code{\link{harmony_options}}.}
  \item{reduction.use, reduction}{Name of the input reduction. Default
    \code{"pca"}.}
  \item{reduction.save, reduction.name}{Name of the written reduction.}
  \item{dims.use}{Which PCs to correct.}
  \item{project.dim}{Project Harmony loadings (Seurat). Default \code{TRUE}.}
  \item{...}{Must be empty (legacy Harmony args are rejected).}
}
\value{
  For the default method, a numeric matrix of corrected embeddings, or a
  Harmony-like object when \code{return_object = TRUE}. Seurat and
  SingleCellExperiment methods return the input object with a Harmony
  reduction added.
}
\examples{
set.seed(1)
pca_matrix <- cbind(rnorm(20), rnorm(20))
batch <- rep(c("A", "B"), each = 10)
\dontrun{
corrected <- RunHarmony(pca_matrix, batch, nclust = 2, max_iter = 2,
                        verbose = FALSE)
}
}
\seealso{
  \code{\link{harmony_options}}, \code{\link{HarmonyConvergencePlot}}
}
