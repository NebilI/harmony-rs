\name{HarmonyConvergencePlot}
\alias{HarmonyConvergencePlot}
\title{Plot Harmony objective over clustering steps}
\description{
  Same role as the unexported \code{HarmonyConvergencePlot} helper in
  immunogenomics/harmony: visualise \code{objective_kmeans} across outer
  rounds.
}
\usage{
HarmonyConvergencePlot(
  harmonyObj,
  round_start = 1,
  round_end = Inf,
  do_wrap = FALSE
)
}
\arguments{
  \item{harmonyObj}{Object from \code{RunHarmony(..., return_object = TRUE)}.}
  \item{round_start}{First outer Harmony round to include (1-based).}
  \item{round_end}{Last outer Harmony round to include.}
  \item{do_wrap}{Facet by outer round when \code{TRUE}.}
}
\value{
  A ggplot object.
}
\examples{
\dontrun{
obj <- RunHarmony(pca_matrix, meta_data, vars_use = "donor",
                  return_object = TRUE)
HarmonyConvergencePlot(obj)
}
}
\seealso{
  \code{\link{RunHarmony}}
}
