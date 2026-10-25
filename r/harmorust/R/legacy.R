#' Reject legacy Harmony arguments with the same errors as immunogenomics/harmony.
#' @keywords internal
.check_legacy_args <- function(...) {
  all.args <- list(...)
  legarg <- c(
    "do_pca", "npcs", "tau", "block.size", "max.iter.harmony",
    "max.iter.cluster", "epsilon.cluster", "epsilon.harmony"
  )
  for (arg in names(all.args)) {
    if (arg %in% legarg) {
      .legacy_error(arg)
    }
  }
  if (length(all.args) > 0) {
    stop(
      paste(
        "Argument", names(all.args),
        "is unhandled. Please refer to the documentation for the valid harmony options!\n"
      ),
      call. = FALSE
    )
  }
  invisible()
}

#' @keywords internal
.legacy_error <- function(param) {
  common_warn <- paste0(
    "Error: The parameter ", param, " has been dropped from the RunHarmony API. ",
    "It will be ignored for this function call ",
    "and please remove parameter ", param, " in future function calls. ",
    "Advanced users can set value of parameter ", param,
    " by using parameter .options and function harmony_options()."
  )
  do_pca_npcs_warn <- paste0(
    "Error: The parameters ", "do_pca and npcs",
    " have been dropped from the RunHarmony API. ",
    "They will be ignored for this function call ",
    "and please remove parameters ", "do_pca and npcs",
    " and pass to harmony cell_embeddings directly."
  )
  max.iter.harmony_warn <- paste0(
    "Error: The parameter ", "max.iter.harmony ",
    "is replaced with parameter ", "max_iter. ",
    "It will be ignored for this function call ",
    "and please use parameter ", "max_iter ",
    "in future function calls."
  )
  epsilon.harmony_warn <- paste0(
    "Error: The parameter ", "epsilon.harmony",
    " has been dropped from the RunHarmony API. ",
    "It will be ignored for this function call ",
    "and please remove parameter ", "epsilon.harmony",
    " in future function calls. ",
    "If users want to control if harmony would stop early or not, ",
    "use parameter ", "early_stop. ",
    "Advanced users can set value of parameter ",
    "epsilon.harmony",
    " by using parameter .options and function harmony_options()."
  )
  warn_str <- common_warn
  if (param %in% c("do_pca", "npcs")) {
    warn_str <- do_pca_npcs_warn
  }
  if (param == "max.iter.harmony") {
    warn_str <- max.iter.harmony_warn
  }
  if (param == "epsilon.harmony") {
    warn_str <- epsilon.harmony_warn
  }
  stop(warn_str, call. = FALSE)
}
