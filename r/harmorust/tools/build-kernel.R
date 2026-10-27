# Build the harmorust CLI during R CMD INSTALL / configure.
# Called from configure and configure.win.

src_file <- function(...) {
  file.path(...)
}

pkg_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
prepare <- src_file(pkg_root, "tools", "prepare-rust.sh")
if (!file.exists(prepare)) {
  stop("tools/prepare-rust.sh is missing; cannot stage the Rust crate")
}

# shell=TRUE so the shebang runs on Windows (Rtools / Git bash) and Unix.
status <- system2("sh", args = prepare, stdout = TRUE, stderr = TRUE)
if (!is.null(attr(status, "status")) && attr(status, "status") != 0) {
  stop(paste(c("prepare-rust.sh failed:", status), collapse = "\n"))
}
if (length(status)) {
  message(paste(status, collapse = "\n"))
}

find_cargo <- function() {
  cargo <- Sys.which("cargo")
  if (nzchar(cargo)) {
    return(cargo)
  }
  exe <- if (.Platform$OS.type == "windows") "cargo.exe" else "cargo"
  candidates <- c(
    file.path(Sys.getenv("HOME"), ".cargo", "bin", exe),
    file.path(Sys.getenv("USERPROFILE"), ".cargo", "bin", exe),
    file.path(Sys.getenv("CARGO_HOME"), "bin", exe),
    "/usr/local/cargo/bin/cargo"
  )
  for (path in candidates) {
    if (nzchar(path) && file.exists(path)) {
      return(path)
    }
  }
  stop(
    "Cargo was not found. Install a Rust toolchain (rustc >= 1.89) from ",
    "https://rustup.rs and retry. harmorust builds its CLI at install time.",
    call. = FALSE
  )
}

cargo <- find_cargo()
manifest <- src_file(pkg_root, "tools", "rust", "Cargo.toml")
if (!file.exists(manifest)) {
  stop("tools/rust/Cargo.toml missing after prepare-rust.sh")
}

message("Building harmorust CLI with ", cargo)
build_status <- system2(
  cargo,
  args = c("build", "--release", "--manifest-path", manifest, "--bin", "harmorust"),
  env = c(paste0("CARGO=", cargo))
)
if (!identical(as.integer(build_status), 0L)) {
  stop("cargo build --release --bin harmorust failed with status ", build_status)
}

bin_name <- if (.Platform$OS.type == "windows") "harmorust.exe" else "harmorust"
built <- src_file(pkg_root, "tools", "rust", "target", "release", bin_name)
if (!file.exists(built)) {
  # cargo may place the binary under target/<triple>/release on some hosts
  alt <- list.files(
    src_file(pkg_root, "tools", "rust", "target"),
    pattern = paste0("^", bin_name, "$"),
    recursive = TRUE,
    full.names = TRUE
  )
  alt <- alt[grepl(file.path("release", bin_name), alt, fixed = TRUE)]
  if (length(alt)) {
    built <- alt[[1]]
  }
}
if (!file.exists(built)) {
  stop("cargo finished but ", bin_name, " was not found under rust/target")
}

dest_dir <- src_file(pkg_root, "inst", "bin")
dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
dest <- src_file(dest_dir, bin_name)
ok <- file.copy(built, dest, overwrite = TRUE)
if (!isTRUE(ok)) {
  stop("failed to copy ", built, " -> ", dest)
}
if (.Platform$OS.type != "windows") {
  Sys.chmod(dest, mode = "0755")
}
message("Installed CLI at ", dest)
