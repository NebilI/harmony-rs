has_kernel <- function() {
  tryCatch(
    {
      .find_harmorust_bin()
      TRUE
    },
    error = function(e) FALSE
  )
}

test_that("installed package ships a harmorust CLI", {
  skip_if_not(has_kernel(), "harmorust CLI is not on PATH or in the package bin")
  bin <- .find_harmorust_bin()
  expect_true(file.exists(bin))
})

test_that("RunHarmony returns a same-shaped embedding", {
  skip_if_not(has_kernel(), "harmorust CLI is not available")
  set.seed(1)
  n <- 40L
  pcs <- cbind(rnorm(n), rnorm(n))
  batch <- rep(c("A", "B"), each = 20L)
  pcs[batch == "B", 1] <- pcs[batch == "B", 1] + 2
  out <- RunHarmony(
    pcs,
    batch,
    nclust = 2,
    max_iter = 3,
    early_stop = FALSE,
    verbose = FALSE
  )
  expect_equal(dim(out), dim(pcs))
  expect_false(anyNA(out))
})
