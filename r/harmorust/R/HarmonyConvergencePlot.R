utils::globalVariables(c("idx", "val", "harmony_idx"))

#' Plot Harmony objective over clustering steps
#'
#' Same role as the unexported `HarmonyConvergencePlot` helper in
#' immunogenomics/harmony: visualise `objective_kmeans` across outer rounds.
#'
#' @param harmonyObj Object from `RunHarmony(..., return_object = TRUE)`.
#' @param round_start First outer Harmony round to include (1-based).
#' @param round_end Last outer Harmony round to include.
#' @param do_wrap Facet by outer round when TRUE.
#' @return A ggplot object.
#' @export
HarmonyConvergencePlot <- function(
    harmonyObj,
    round_start = 1,
    round_end = Inf,
    do_wrap = FALSE
) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("HarmonyConvergencePlot requires ggplot2")
  }
  if (is.null(harmonyObj$kmeans_rounds) || is.null(harmonyObj$objective_kmeans)) {
    stop("harmonyObj must contain kmeans_rounds and objective_kmeans")
  }
  rounds <- as.integer(harmonyObj$kmeans_rounds)
  if (!length(rounds) || any(rounds < 1L)) {
    stop("kmeans_rounds must contain positive iteration counts")
  }
  vals <- utils::tail(as.numeric(harmonyObj$objective_kmeans), -1)
  if (length(vals) != sum(rounds)) {
    stop("objective_kmeans length does not match kmeans_rounds")
  }

  obj_fxn <- data.frame(
    idx = seq_along(vals),
    kmeans_idx = unlist(lapply(rounds, seq_len), use.names = FALSE),
    harmony_idx = unlist(lapply(seq_along(rounds), function(i) rep(i, rounds[i])), use.names = FALSE),
    val = vals,
    stringsAsFactors = FALSE
  )
  obj_fxn <- obj_fxn[
    obj_fxn$harmony_idx >= round_start & obj_fxn$harmony_idx <= round_end,
    ,
    drop = FALSE
  ]
  obj_fxn$idx <- seq_len(nrow(obj_fxn))

  plt <- ggplot2::ggplot(
    obj_fxn,
    ggplot2::aes(
      x = idx,
      y = val,
      colour = as.factor(harmony_idx)
    )
  ) +
    ggplot2::geom_point() +
    ggplot2::labs(
      y = "Objective Function",
      x = "Clustering Step #",
      colour = "Integration #"
    )
  if (isTRUE(do_wrap)) {
    plt <- plt + ggplot2::facet_grid(~harmony_idx, scales = "free", space = "free_x")
  }
  plt
}
