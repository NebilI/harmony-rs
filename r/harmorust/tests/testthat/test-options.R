test_that("harmony_options returns the expected class", {
  opts <- harmony_options()
  expect_s3_class(opts, "harmony_options")
  expect_equal(opts$alpha, 0.2)
  expect_equal(opts$block.size, 0.05)
})

test_that("harmony_options rejects an invalid block.size", {
  expect_error(harmony_options(block.size = 0), "block.size")
  expect_error(harmony_options(block.size = 1.2), "block.size")
})

test_that("legacy Harmony arguments are rejected", {
  expect_error(
    .check_legacy_args(do_pca = TRUE),
    "do_pca and npcs"
  )
  expect_error(
    .check_legacy_args(max.iter.harmony = 10),
    "max_iter"
  )
})
