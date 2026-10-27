\name{harmony_options}
\alias{harmony_options}
\title{Advanced Harmony options}
\description{
  Same names as \code{harmony::harmony_options}.
}
\usage{
harmony_options(
  alpha = 0.2,
  tau = 0,
  block.size = 0.05,
  max.iter.cluster = 4,
  epsilon.cluster = 1e-3,
  epsilon.harmony = 1e-2,
  batch.prop.cutoff = 1e-5
)
}
\arguments{
  \item{alpha}{Harmony2 automatic-lambda scale. Default 0.2.}
  \item{tau}{Softens theta for small batches. Default 0.}
  \item{block.size}{Fraction of cells updated per clustering block.
    \code{(0, 1]}.}
  \item{max.iter.cluster}{Inner clustering iterations per Harmony round.}
  \item{epsilon.cluster}{Clustering convergence tolerance.}
  \item{epsilon.harmony}{Outer-loop convergence tolerance.}
  \item{batch.prop.cutoff}{Skip a batch in a cluster below this soft
    membership.}
}
\value{
  An object of class \code{harmony_options}.
}
\examples{
opts <- harmony_options(block.size = 0.1, max.iter.cluster = 5)
}
