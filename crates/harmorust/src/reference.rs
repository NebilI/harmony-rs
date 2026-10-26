//! Slow, direct transliteration of the Harmony float ops.
//!
//! Production kernels must match these bitwise on random inputs. The
//! functions here exist only for tests — they are not used by `run_harmony`.

use crate::linalg::{accu_sq_two, euclid2_two_acc, invert_arrowhead_f32, softmax_neg_dist};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Armadillo two-accumulator column L2, then divide (Harmony `normalise(., 2, 0)`).
pub fn reference_l2_normalize_columns(z: ArrayView2<'_, f32>) -> Array2<f32> {
    let d = z.nrows();
    let n = z.ncols();
    let mut out = z.to_owned();
    for j in 0..n {
        let norm = if d < 32 {
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut i = 0usize;
            while i + 1 < d {
                acc1 += out[[i, j]] * out[[i, j]];
                acc2 += out[[i + 1, j]] * out[[i + 1, j]];
                i += 2;
            }
            if i < d {
                acc1 += out[[i, j]] * out[[i, j]];
            }
            (acc1 + acc2).sqrt()
        } else {
            let mut s = 0.0f32;
            for i in 0..d {
                s += out[[i, j]] * out[[i, j]];
            }
            s.sqrt()
        };
        if norm > 0.0 {
            for i in 0..d {
                out[[i, j]] /= norm;
            }
        }
    }
    out
}

/// Harmony init softmax: `R = exp(-dist / sigma)` then two-accumulator column L1.
pub fn reference_softmax_neg_dist(dist: ArrayView2<'_, f32>, sigma: ArrayView1<'_, f32>) -> Array2<f32> {
    let k = dist.nrows();
    let n = dist.ncols();
    let mut r = Array2::<f32>::zeros((k, n));
    for i in 0..k {
        let sig = sigma[i];
        for j in 0..n {
            r[[i, j]] = (-dist[[i, j]] / sig).exp();
        }
    }
    for j in 0..n {
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut i = 0usize;
        while i + 1 < k {
            acc1 += r[[i, j]];
            acc2 += r[[i + 1, j]];
            i += 2;
        }
        if i < k {
            acc1 += r[[i, j]];
        }
        let s = acc1 + acc2;
        if s > 0.0 {
            for i in 0..k {
                r[[i, j]] /= s;
            }
        }
    }
    r
}

/// Sequential `my_accu` of `R % dist` in cell-then-cluster (column-major) order.
#[allow(dead_code)]
pub fn reference_my_accu_rdist(r: ArrayView2<'_, f32>, dist: ArrayView2<'_, f32>) -> f32 {
    let mut acc = 0.0f32;
    for j in 0..r.ncols() {
        for i in 0..r.nrows() {
            acc += r[[i, j]] * dist[[i, j]];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{l2_normalize_columns, softmax_neg_dist};
    use ndarray::{Array1, Array2};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn random_matrix(d: usize, n: usize, seed: u64) -> Array2<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((d, n), |_| rng.gen::<f32>() - 0.5)
    }

    #[test]
    fn softmax_matches_reference_bitwise() {
        let dist = random_matrix(7, 31, 3).mapv(|v| v.abs() + 0.01);
        let sigma = Array1::from_elem(7, 0.1f32);
        let want = reference_softmax_neg_dist(dist.view(), sigma.view());
        let dist64 = dist.mapv(f64::from);
        let sigma64 = sigma.mapv(f64::from);
        let got = softmax_neg_dist(dist64.view(), sigma64.view());
        for i in 0..want.nrows() {
            for j in 0..want.ncols() {
                let g = got[[i, j]] as f32;
                assert_eq!(
                    g.to_bits(),
                    want[[i, j]].to_bits(),
                    "softmax [{i},{j}] got {g} want {}",
                    want[[i, j]]
                );
            }
        }
    }

    #[test]
    fn l2_matches_reference_bitwise() {
        let z = random_matrix(8, 17, 11);
        let want = reference_l2_normalize_columns(z.view());
        let got = l2_normalize_columns(z.mapv(f64::from).view());
        for i in 0..want.nrows() {
            for j in 0..want.ncols() {
                let g = got[[i, j]] as f32;
                assert_eq!(
                    g.to_bits(),
                    want[[i, j]].to_bits(),
                    "l2 [{i},{j}] got {g} want {}",
                    want[[i, j]]
                );
            }
        }
    }

    #[test]
    fn euclid_two_acc_matches_scalar_pair() {
        let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.3 - 1.0).collect();
        let b: Vec<f32> = (0..19).map(|i| i as f32 * 0.1 + 0.2).collect();
        let got = euclid2_two_acc(|i| a[i], |i| b[i], 19);
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut i = 0usize;
        let mut j = 1usize;
        while j < 19 {
            let ti = a[i] - b[i];
            let tj = a[j] - b[j];
            acc1 += ti * ti;
            acc2 += tj * tj;
            i += 2;
            j += 2;
        }
        if i < 19 {
            let ti = a[i] - b[i];
            acc1 += ti * ti;
        }
        assert_eq!(got.to_bits(), (acc1 + acc2).to_bits());
    }

    #[test]
    fn arrowhead_inverse_is_self_consistent() {
        let mut cov = Array2::<f32>::zeros((4, 4));
        cov[[0, 0]] = 10.0;
        for i in 1..4 {
            cov[[0, i]] = 2.0 + i as f32;
            cov[[i, 0]] = 2.0 + i as f32;
            cov[[i, i]] = 3.0 + i as f32;
        }
        let inv = invert_arrowhead_f32(cov.view()).unwrap();
        let prod = cov.dot(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((prod[[i, j]] - want).abs() < 2e-5, "I [{i},{j}] = {}", prod[[i, j]]);
            }
        }
    }

    #[test]
    fn accu_sq_two_matches_hand_fold() {
        let xs = [0.5f32, -1.0, 2.0, 0.25, 3.0];
        let got = accu_sq_two(xs);
        let want = (0.5f32 * 0.5 + 2.0 * 2.0 + 3.0 * 3.0) + (1.0 + 0.25 * 0.25);
        assert_eq!(got.to_bits(), want.to_bits());
        let _ = reference_my_accu_rdist;
        let _ = softmax_neg_dist;
    }
}
