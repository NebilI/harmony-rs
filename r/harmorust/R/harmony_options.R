#' Advanced Harmony options (same names as `harmony::harmony_options`).
#'
#' @param alpha Harmony2 automatic-lambda scale. Default 0.2.
#' @param tau Softens theta for small batches. Default 0.
#' @param block.size Fraction of cells updated per clustering block. `(0, 1]`.
#' @param max.iter.cluster Inner clustering iterations per Harmony round.
#' @param epsilon.cluster Clustering convergence tolerance.
#' @param epsilon.harmony Outer-loop convergence tolerance.
#' @param batch.prop.cutoff Skip a batch in a cluster below this soft membership.
#' @return An object of class `harmony_options`.
#' @export
harmony_options <- function(
    alpha = 0.2,
    tau = 0,
    block.size = 0.05,
    max.iter.cluster = 4,
    epsilon.cluster = 1e-3,
    epsilon.harmony = 1e-2,
    batch.prop.cutoff = 1e-5
) {
  if (block.size <= 0 || block.size > 1) {
    stop("Error: block.size should be set between 0 and 1 (0 < block.size <= 1)")
  }
  structure(
    list(
      alpha = alpha,
      tau = tau,
      block.size = block.size,
      max.iter.cluster = as.integer(max.iter.cluster),
      epsilon.cluster = epsilon.cluster,
      epsilon.harmony = epsilon.harmony,
      batch.prop.cutoff = batch.prop.cutoff
    ),
    class = "harmony_options"
  )
}
