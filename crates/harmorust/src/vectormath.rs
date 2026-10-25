//! Slice-wise `exp` / `ln` using glibc libmvec when available.
//!
//! Softmax and the Harmony objective walk whole rows of a `K × N` matrix.
//! Those rows are contiguous, so a 4- or 8-wide libm call is the natural
//! replacement for scalar `libm` `exp`/`ln`. On non-Linux x86_64 we fall
//! back to the same scalar libm the old loops used.

#[cfg(harmorust_libmvec)]
extern "C" {
    fn harmorust_exp_avx2(dst: *mut f64, src: *const f64, n: usize);
    fn harmorust_ln_nonpos_zero_avx2(dst: *mut f64, src: *const f64, n: usize);
    fn harmorust_exp_avx512(dst: *mut f64, src: *const f64, n: usize);
    fn harmorust_ln_nonpos_zero_avx512(dst: *mut f64, src: *const f64, n: usize);
}

/// `dst[i] = exp(src[i])`.
pub fn exp_slice(dst: &mut [f64], src: &[f64]) {
    assert_eq!(dst.len(), src.len());
    let n = dst.len();
    if n == 0 {
        return;
    }
    #[cfg(harmorust_libmvec)]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                harmorust_exp_avx512(dst.as_mut_ptr(), src.as_ptr(), n);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe {
                harmorust_exp_avx2(dst.as_mut_ptr(), src.as_ptr(), n);
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
    #[cfg(harmorust_libmvec)]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                harmorust_ln_nonpos_zero_avx512(dst.as_mut_ptr(), src.as_ptr(), n);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe {
                harmorust_ln_nonpos_zero_avx2(dst.as_mut_ptr(), src.as_ptr(), n);
            }
            return;
        }
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = if *s > 0.0 { s.ln() } else { 0.0 };
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
