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
crate_dir <- src_file(pkg_root, "tools", "rust")
manifest <- src_file(crate_dir, "Cargo.toml")
if (!file.exists(manifest)) {
  stop("tools/rust/Cargo.toml missing after prepare-rust.sh")
}

# Do not pass env= to system2(). On Windows that is forwarded as extra cargo
# arguments (`no such subcommand CARGO=...`) and r-universe Windows builds fail.
# Run from the crate directory so wrappers such as rwasm's cargo can find
# ./target after the build (they cd into it relative to the working directory).
oldwd <- getwd()
on.exit(setwd(oldwd), add = TRUE)
setwd(crate_dir)

message("Building harmorust CLI with ", cargo)
rustc <- Sys.which("rustc")
if (!nzchar(rustc)) {
  rustc_candidates <- c(
    file.path(dirname(cargo), if (.Platform$OS.type == "windows") "rustc.exe" else "rustc")
  )
  rustc <- rustc_candidates[[1]]
}
if (nzchar(rustc) && file.exists(rustc)) {
  message(paste(system2(rustc, args = "--version", stdout = TRUE), collapse = " "))
}
build_status <- system2(
  cargo,
  args = c("build", "--release", "--bin", "harmorust")
)
if (!identical(as.integer(build_status), 0L)) {
  stop("cargo build --release --bin harmorust failed with status ", build_status)
}

bin_name <- if (.Platform$OS.type == "windows") "harmorust.exe" else "harmorust"
built <- src_file(crate_dir, "target", "release", bin_name)
if (!file.exists(built)) {
  # cargo / rwasm may place the binary under target/<triple>/release
  alt <- list.files(
    src_file(crate_dir, "target"),
    pattern = paste0("^", bin_name, "$"),
    recursive = TRUE,
    full.names = TRUE
  )
  alt <- alt[grepl(paste0("release", .Platform$file.sep, bin_name, "$"), alt)]
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
