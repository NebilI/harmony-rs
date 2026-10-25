fn main() {
    println!("cargo:rustc-check-cfg=cfg(harmorust_libmvec)");
    if !(cfg!(target_arch = "x86_64") && cfg!(target_os = "linux")) {
        return;
    }

    println!("cargo:rerun-if-changed=src/vectormath_avx2.c");
    println!("cargo:rerun-if-changed=src/vectormath_avx512.c");

    cc::Build::new()
        .file("src/vectormath_avx2.c")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx2")
        .compile("harmorust_vectormath_avx2");

    cc::Build::new()
        .file("src/vectormath_avx512.c")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx2")
        .flag_if_supported("-mavx512f")
        .compile("harmorust_vectormath_avx512");

    println!("cargo:rustc-link-lib=mvec");
    println!("cargo:rustc-cfg=harmorust_libmvec");
}
