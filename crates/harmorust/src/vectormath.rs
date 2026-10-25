//! Slice-wise `exp` / `ln` using `std::arch` (AVX2 / AVX-512).
//!
//! Softmax and the Harmony objective walk whole rows of a `K × N` matrix.
//! Those rows are contiguous, so a 4- or 8-wide polynomial replaces scalar
//! `libm` `exp`/`ln`. Requires rustc ≥ 1.89 for stable AVX-512.
//!
//! glibc libmvec was the first landing, but calling it from Rust needs
//! nightly `simd_ffi`. This module is pure Rust.

/// `dst[i] = exp(src[i])`.
pub fn exp_slice(dst: &mut [f64], src: &[f64]) {
    assert_eq!(dst.len(), src.len());
    let n = dst.len();
    if n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                x86::exp_avx512(dst, src);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                x86::exp_avx2(dst, src);
            }
            return;
        }
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.exp();
    }
}

/// `dst[i] = ln(src[i])` when `src[i] > 0`, else `0`. Matches `safe_log`.
pub fn ln_nonpositive_zero(dst: &mut [f64], src: &[f64]) {
    assert_eq!(dst.len(), src.len());
    let n = dst.len();
    if n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                x86::ln_avx512(dst, src);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                x86::ln_avx2(dst, src);
            }
            return;
        }
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = if *s > 0.0 { s.ln() } else { 0.0 };
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    const INV_LN2: f64 = 1.442_695_040_888_963_387_00;
    const SQRT2: f64 = 1.414_213_562_373_095_1;

    const EXP_P0: f64 = 1.261_771_930_748_105_908_78e-4;
    const EXP_P1: f64 = 3.029_944_077_074_419_613_00e-2;
    const EXP_P2: f64 = 1.0;
    const EXP_Q0: f64 = 3.001_985_051_386_644_550_42e-6;
    const EXP_Q1: f64 = 2.524_483_403_496_841_041_92e-3;
    const EXP_Q2: f64 = 2.272_655_482_081_550_287_66e-1;
    const EXP_Q3: f64 = 2.0;

    const LG1: f64 = 6.666_666_666_666_735_130e-1;
    const LG2: f64 = 3.999_999_999_940_941_908e-1;
    const LG3: f64 = 2.857_142_874_366_239_149e-1;
    const LG4: f64 = 2.222_219_843_214_978_396e-1;
    const LG5: f64 = 1.818_357_216_161_805_012e-1;
    const LG6: f64 = 1.531_383_769_920_937_332e-1;
    const LG7: f64 = 1.479_819_860_511_658_591e-1;

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn exp_avx2(dst: &mut [f64], src: &[f64]) {
        let n = src.len();
        let mut i = 0;
        while i + 4 <= n {
            let x = _mm256_loadu_pd(src.as_ptr().add(i));
            _mm256_storeu_pd(dst.as_mut_ptr().add(i), exp4(x));
            i += 4;
        }
        while i < n {
            dst[i] = src[i].exp();
            i += 1;
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn ln_avx2(dst: &mut [f64], src: &[f64]) {
        let n = src.len();
        let mut i = 0;
        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);
        while i + 4 <= n {
            let x = _mm256_loadu_pd(src.as_ptr().add(i));
            let pos = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);
            let safe = _mm256_blendv_pd(one, x, pos);
            let y = _mm256_and_pd(log4(safe), pos);
            _mm256_storeu_pd(dst.as_mut_ptr().add(i), y);
            i += 4;
        }
        while i < n {
            dst[i] = if src[i] > 0.0 { src[i].ln() } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn exp_avx512(dst: &mut [f64], src: &[f64]) {
        let n = src.len();
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm512_loadu_pd(src.as_ptr().add(i));
            _mm512_storeu_pd(dst.as_mut_ptr().add(i), exp8(x));
            i += 8;
        }
        while i < n {
            dst[i] = src[i].exp();
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn ln_avx512(dst: &mut [f64], src: &[f64]) {
        let n = src.len();
        let mut i = 0;
        let zero = _mm512_setzero_pd();
        let one = _mm512_set1_pd(1.0);
        while i + 8 <= n {
            let x = _mm512_loadu_pd(src.as_ptr().add(i));
            let pos = _mm512_cmp_pd_mask(x, zero, _CMP_GT_OQ);
            let safe = _mm512_mask_blend_pd(pos, one, x);
            let y = _mm512_maskz_mov_pd(pos, log8(safe));
            _mm512_storeu_pd(dst.as_mut_ptr().add(i), y);
            i += 8;
        }
        while i < n {
            dst[i] = if src[i] > 0.0 { src[i].ln() } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn exp4(x: __m256d) -> __m256d {
        let x = _mm256_min_pd(x, _mm256_set1_pd(709.0));
        let x = _mm256_max_pd(x, _mm256_set1_pd(-709.0));
        let n = _mm256_round_pd(
            _mm256_mul_pd(x, _mm256_set1_pd(INV_LN2)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let r = _mm256_sub_pd(
            _mm256_sub_pd(x, _mm256_mul_pd(n, _mm256_set1_pd(LN2_HI))),
            _mm256_mul_pd(n, _mm256_set1_pd(LN2_LO)),
        );
        let xx = _mm256_mul_pd(r, r);
        let px = _mm256_mul_pd(
            r,
            _mm256_fmadd_pd(
                xx,
                _mm256_fmadd_pd(xx, _mm256_set1_pd(EXP_P0), _mm256_set1_pd(EXP_P1)),
                _mm256_set1_pd(EXP_P2),
            ),
        );
        let qx = _mm256_fmadd_pd(
            xx,
            _mm256_fmadd_pd(
                xx,
                _mm256_fmadd_pd(xx, _mm256_set1_pd(EXP_Q0), _mm256_set1_pd(EXP_Q1)),
                _mm256_set1_pd(EXP_Q2),
            ),
            _mm256_set1_pd(EXP_Q3),
        );
        let e = _mm256_fmadd_pd(
            _mm256_set1_pd(2.0),
            _mm256_div_pd(px, _mm256_sub_pd(qx, px)),
            _mm256_set1_pd(1.0),
        );
        _mm256_mul_pd(e, pow2_from_n4(n))
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn log4(x: __m256d) -> __m256d {
        let bits = _mm256_castpd_si256(x);
        let raw_exp = _mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(0x7ff));
        let e_i = _mm256_sub_epi64(raw_exp, _mm256_set1_epi64x(1023));
        let mant = _mm256_or_si256(
            _mm256_and_si256(bits, _mm256_set1_epi64x(0x000f_ffff_ffff_ffff)),
            _mm256_set1_epi64x(0x3ff0_0000_0000_0000),
        );
        let mut m = _mm256_castsi256_pd(mant);
        let mut e = i64_to_f64_4(e_i);
        let big = _mm256_cmp_pd(m, _mm256_set1_pd(SQRT2), _CMP_GE_OQ);
        m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), big);
        e = _mm256_blendv_pd(e, _mm256_add_pd(e, _mm256_set1_pd(1.0)), big);

        let f = _mm256_sub_pd(m, _mm256_set1_pd(1.0));
        let s = _mm256_div_pd(f, _mm256_add_pd(_mm256_set1_pd(2.0), f));
        let z = _mm256_mul_pd(s, s);
        let w = _mm256_mul_pd(z, z);
        let t1 = _mm256_mul_pd(
            w,
            _mm256_fmadd_pd(
                w,
                _mm256_fmadd_pd(w, _mm256_set1_pd(LG6), _mm256_set1_pd(LG4)),
                _mm256_set1_pd(LG2),
            ),
        );
        let t2 = _mm256_mul_pd(
            z,
            _mm256_fmadd_pd(
                w,
                _mm256_fmadd_pd(
                    w,
                    _mm256_fmadd_pd(w, _mm256_set1_pd(LG7), _mm256_set1_pd(LG5)),
                    _mm256_set1_pd(LG3),
                ),
                _mm256_set1_pd(LG1),
            ),
        );
        let r = _mm256_add_pd(t1, t2);
        let hfsq = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(f, f));
        let log_m = _mm256_sub_pd(
            f,
            _mm256_sub_pd(hfsq, _mm256_mul_pd(s, _mm256_add_pd(hfsq, r))),
        );
        _mm256_fmadd_pd(
            e,
            _mm256_set1_pd(LN2_HI),
            _mm256_fmadd_pd(e, _mm256_set1_pd(LN2_LO), log_m),
        )
    }

    #[target_feature(enable = "avx2")]
    unsafe fn pow2_from_n4(n: __m256d) -> __m256d {
        let n32 = _mm256_cvtpd_epi32(n);
        let n64 = _mm256_cvtepi32_epi64(n32);
        let biased = _mm256_add_epi64(n64, _mm256_set1_epi64x(1023));
        _mm256_castsi256_pd(_mm256_slli_epi64(biased, 52))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn i64_to_f64_4(v: __m256i) -> __m256d {
        let packed = _mm256_shuffle_epi32(v, 0x88);
        let packed = _mm256_permute4x64_epi64(packed, 0x08);
        _mm256_cvtepi32_pd(_mm256_castsi256_si128(packed))
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn exp8(x: __m512d) -> __m512d {
        let x = _mm512_min_pd(x, _mm512_set1_pd(709.0));
        let x = _mm512_max_pd(x, _mm512_set1_pd(-709.0));
        let n = _mm512_roundscale_pd(_mm512_mul_pd(x, _mm512_set1_pd(INV_LN2)), 0);
        let r = _mm512_sub_pd(
            _mm512_sub_pd(x, _mm512_mul_pd(n, _mm512_set1_pd(LN2_HI))),
            _mm512_mul_pd(n, _mm512_set1_pd(LN2_LO)),
        );
        let xx = _mm512_mul_pd(r, r);
        let px = _mm512_mul_pd(
            r,
            _mm512_fmadd_pd(
                xx,
                _mm512_fmadd_pd(xx, _mm512_set1_pd(EXP_P0), _mm512_set1_pd(EXP_P1)),
                _mm512_set1_pd(EXP_P2),
            ),
        );
        let qx = _mm512_fmadd_pd(
            xx,
            _mm512_fmadd_pd(
                xx,
                _mm512_fmadd_pd(xx, _mm512_set1_pd(EXP_Q0), _mm512_set1_pd(EXP_Q1)),
                _mm512_set1_pd(EXP_Q2),
            ),
            _mm512_set1_pd(EXP_Q3),
        );
        let e = _mm512_fmadd_pd(
            _mm512_set1_pd(2.0),
            _mm512_div_pd(px, _mm512_sub_pd(qx, px)),
            _mm512_set1_pd(1.0),
        );
        _mm512_mul_pd(e, pow2_from_n8(n))
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn log8(x: __m512d) -> __m512d {
        let bits = _mm512_castpd_si512(x);
        let raw_exp = _mm512_and_si512(_mm512_srli_epi64(bits, 52), _mm512_set1_epi64(0x7ff));
        let e_i = _mm512_sub_epi64(raw_exp, _mm512_set1_epi64(1023));
        let mant = _mm512_or_si512(
            _mm512_and_si512(bits, _mm512_set1_epi64(0x000f_ffff_ffff_ffff)),
            _mm512_set1_epi64(0x3ff0_0000_0000_0000),
        );
        let mut m = _mm512_castsi512_pd(mant);
        let mut e = _mm512_cvtepi32_pd(_mm512_cvtepi64_epi32(e_i));
        let big = _mm512_cmp_pd_mask(m, _mm512_set1_pd(SQRT2), _CMP_GE_OQ);
        m = _mm512_mask_mul_pd(m, big, m, _mm512_set1_pd(0.5));
        e = _mm512_mask_add_pd(e, big, e, _mm512_set1_pd(1.0));

        let f = _mm512_sub_pd(m, _mm512_set1_pd(1.0));
        let s = _mm512_div_pd(f, _mm512_add_pd(_mm512_set1_pd(2.0), f));
        let z = _mm512_mul_pd(s, s);
        let w = _mm512_mul_pd(z, z);
        let t1 = _mm512_mul_pd(
            w,
            _mm512_fmadd_pd(
                w,
                _mm512_fmadd_pd(w, _mm512_set1_pd(LG6), _mm512_set1_pd(LG4)),
                _mm512_set1_pd(LG2),
            ),
        );
        let t2 = _mm512_mul_pd(
            z,
            _mm512_fmadd_pd(
                w,
                _mm512_fmadd_pd(
                    w,
                    _mm512_fmadd_pd(w, _mm512_set1_pd(LG7), _mm512_set1_pd(LG5)),
                    _mm512_set1_pd(LG3),
                ),
                _mm512_set1_pd(LG1),
            ),
        );
        let r = _mm512_add_pd(t1, t2);
        let hfsq = _mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(f, f));
        let log_m = _mm512_sub_pd(
            f,
            _mm512_sub_pd(hfsq, _mm512_mul_pd(s, _mm512_add_pd(hfsq, r))),
        );
        _mm512_fmadd_pd(
            e,
            _mm512_set1_pd(LN2_HI),
            _mm512_fmadd_pd(e, _mm512_set1_pd(LN2_LO), log_m),
        )
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn pow2_from_n8(n: __m512d) -> __m512d {
        let n32 = _mm512_cvttpd_epi32(n);
        let n64 = _mm512_cvtepi32_epi64(n32);
        let biased = _mm512_add_epi64(n64, _mm512_set1_epi64(1023));
        _mm512_castsi512_pd(_mm512_slli_epi64(biased, 52))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(got: &[f64], want: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn exp_matches_libm_on_softmax_range() {
        let src: Vec<f64> = (0..257)
            .map(|i| -40.0 + i as f64 * 40.0 / 256.0)
            .collect();
        let mut got = vec![0.0; src.len()];
        let want: Vec<f64> = src.iter().map(|x| x.exp()).collect();
        exp_slice(&mut got, &src);
        let err = max_abs_err(&got, &want);
        assert!(err < 1e-12, "max |Δexp| = {err}");
    }

    #[test]
    fn ln_matches_libm_on_assignment_range() {
        let src: Vec<f64> = (0..257)
            .map(|i| 1e-16 * (1e16_f64).powf(i as f64 / 256.0))
            .collect();
        let mut got = vec![0.0; src.len()];
        let want: Vec<f64> = src.iter().map(|x| x.ln()).collect();
        ln_nonpositive_zero(&mut got, &src);
        let err = max_abs_err(&got, &want);
        assert!(err < 1e-12, "max |Δln| = {err}");
    }

    #[test]
    fn ln_maps_nonpositive_to_zero() {
        let src = [0.0, -1.0, 1.0, -0.0, 0.5];
        let mut got = [0.0; 5];
        ln_nonpositive_zero(&mut got, &src);
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], 0.0);
        assert!((got[2] - 0.0).abs() < 1e-15);
        assert_eq!(got[3], 0.0);
        assert!((got[4] - 0.5_f64.ln()).abs() < 1e-14);
    }
}
