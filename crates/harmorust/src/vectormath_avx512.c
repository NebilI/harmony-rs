#include <immintrin.h>
#include <math.h>
#include <stddef.h>

/* glibc libmvec, AVX-512 8×f64 and AVX2 4×f64 tail. */
extern __m512d _ZGVeN8v_exp(__m512d);
extern __m512d _ZGVeN8v_log(__m512d);
extern __m256d _ZGVdN4v_exp(__m256d);
extern __m256d _ZGVdN4v_log(__m256d);

void harmorust_exp_avx512(double *dst, const double *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, _ZGVeN8v_exp(x));
    }
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, _ZGVdN4v_exp(x));
    }
    for (; i < n; i++) {
        dst[i] = exp(src[i]);
    }
}

void harmorust_ln_nonpos_zero_avx512(double *dst, const double *src, size_t n) {
    size_t i = 0;
    const __m512d zero512 = _mm512_setzero_pd();
    const __m512d one512 = _mm512_set1_pd(1.0);
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(src + i);
        __mmask8 pos = _mm512_cmp_pd_mask(x, zero512, _CMP_GT_OQ);
        __m512d safe = _mm512_mask_blend_pd(pos, one512, x);
        __m512d y = _ZGVeN8v_log(safe);
        _mm512_storeu_pd(dst + i, _mm512_maskz_mov_pd(pos, y));
    }
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(src + i);
        __m256d pos = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);
        __m256d safe = _mm256_blendv_pd(one, x, pos);
        __m256d y = _ZGVdN4v_log(safe);
        _mm256_storeu_pd(dst + i, _mm256_and_pd(y, pos));
    }
    for (; i < n; i++) {
        dst[i] = src[i] > 0.0 ? log(src[i]) : 0.0;
    }
}
