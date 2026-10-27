\name{harmorust-package}
\alias{harmorust-package}
\alias{harmorust}
\docType{package}
\title{Rust Native Kernels for Harmony Single-Cell Integration}
\description{
  Rust reimplementation of the Harmony integration loop with the same
  \code{RunHarmony()} signature used by the immunogenomics Harmony R package.
  A source install compiles the \code{harmorust} CLI with Cargo and stores it
  in the package \file{bin} directory.
}
\details{
  \tabular{ll}{
    Package: \tab harmorust\cr
    Version: \tab 0.1.0\cr
    License: \tab MIT\cr
  }
  Typical workflow: Seurat or seurust produce a PCA embedding, then
  \code{\link{RunHarmony}} corrects batch effects.
}
\author{
  Nebil Ibrahim \email{nebil.b.ibrahim@gmail.com}
}
\seealso{
  \code{\link{RunHarmony}}, \code{\link{harmony_options}},
  \code{\link{HarmonyConvergencePlot}}
}
\keyword{package}
