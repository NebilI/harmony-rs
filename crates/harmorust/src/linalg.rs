use crate::error::{HarmonyError, Result};
use crate::vectormath::exp_slice;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

pub fn l2_normalize_columns(z: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut out = z.to_owned();
    for mut col in out.columns_mut() {
        let norm = col.dot(&col).sqrt();
        if norm > 0.0 {
            col.mapv_inplace(|v| v / norm);
        }
    }
    out
}

pub fn cosine_distance(y: ArrayView2<'_, f64>, z: ArrayView2<'_, f64>) -> Array2<f64> {
    // Y is d x K, Z is d x N. Dist = 2 (1 - Y^T Z)
    let sim = y.t().dot(&z);
    sim.mapv(|s| 2.0 * (1.0 - s))
}

pub fn column_l1_normalize(mut r: Array2<f64>) -> Array2<f64> {
    for mut col in r.columns_mut() {
        let s: f64 = col.sum();
        if s > 0.0 {
            col.mapv_inplace(|v| v / s);
        }
    }
    r
}

pub fn invert(a: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(HarmonyError::msg("invert: matrix is not square"));
    }
    let mut m = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            m[[i, j]] = a[[i, j]];
        }
        m[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        let mut pivot = col;
        let mut best = m[[col, col]].abs();
        for r in (col + 1)..n {
            let v = m[[r, col]].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-18 {
            return Err(HarmonyError::msg(
                "ridge covariance is singular; try a larger lambda",
            ));
        }
        if pivot != col {
            for j in 0..(2 * n) {
                let tmp = m[[col, j]];
                m[[col, j]] = m[[pivot, j]];
                m[[pivot, j]] = tmp;
            }
        }
        let diag = m[[col, col]];
        for j in col..(2 * n) {
            m[[col, j]] /= diag;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[[r, col]];
            if f == 0.0 {
                continue;
            }
            for j in col..(2 * n) {
                m[[r, j]] -= f * m[[col, j]];
            }
        }
    }

    Ok(m.slice(s![.., n..]).to_owned())
}

/// Arrowhead inverse used by Harmony2 for a single covariate.
///
/// `phi_cov` is (B+1) x (B+1) with intercept in row/col 0 and a diagonal
/// batch block.
pub fn invert_arrowhead(phi_cov: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
    let n = phi_cov.nrows();
    let mut ac = Array1::<f64>::zeros(n);
    ac[0] = 1.0;
    for i in 1..n {
        ac[i] = -phi_cov[[0, i]];
    }
    let b0 = phi_cov[[0, 0]];
    let mut b = Array1::<f64>::zeros(n);
    for i in 1..n {
        let d = phi_cov[[i, i]];
        if d.abs() < 1e-18 {
            return invert(phi_cov);
        }
        b[i] = 1.0 / d;
    }
    let mut u = b0;
    for i in 1..n {
        u -= ac[i] * ac[i] * b[i];
    }
    if u.abs() < 1e-18 {
        return invert(phi_cov);
    }
    let mut ac_b = Array1::<f64>::zeros(n);
    ac_b[0] = 1.0;
    for i in 1..n {
        ac_b[i] = ac[i] * b[i];
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = ac_b[i] * ac_b[j] / u;
        }
        inv[[i, i]] += b[i];
    }
    Ok(inv)
}

pub fn softmax_neg_dist(dist: ArrayView2<'_, f64>, sigma: ArrayView1<'_, f64>) -> Array2<f64> {
    let k = dist.nrows();
    let n = dist.ncols();
    let mut r = Array2::<f64>::zeros((k, n));
    if n == 0 || k == 0 {
        return r;
    }

    // Rows of the K×N distance matrix are contiguous. Walk cells that way so
    // libmvec can exp a whole row after a per-cell max reduction.
    let mut max_logit = vec![f64::NEG_INFINITY; n];
    for i in 0..k {
        let inv_sigma = 1.0 / sigma[i];
        let drow = dist.row(i);
        for j in 0..n {
            let logit = -drow[j] * inv_sigma;
            if logit > max_logit[j] {
                max_logit[j] = logit;
            }
        }
    }

    let mut scratch = vec![0.0f64; n];
    let mut sum = vec![0.0f64; n];
    for i in 0..k {
        let inv_sigma = 1.0 / sigma[i];
        let drow = dist.row(i);
        for j in 0..n {
            scratch[j] = -drow[j] * inv_sigma - max_logit[j];
        }
        let mut row = r.row_mut(i);
        let rrow = row.as_slice_mut().expect("softmax row is contiguous");
        exp_slice(rrow, &scratch);
        for j in 0..n {
            sum[j] += rrow[j];
        }
    }

    for i in 0..k {
        let mut row = r.row_mut(i);
        let rrow = row.as_slice_mut().expect("softmax row is contiguous");
        for j in 0..n {
            if sum[j] > 0.0 {
                rrow[j] /= sum[j];
            }
        }
    }
    r
}

pub fn safe_log(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        x.ln()
    }
}

pub fn mean_axis1(z: ArrayView2<'_, f64>) -> Array1<f64> {
    z.mean_axis(Axis(1)).unwrap_or_else(|| Array1::zeros(z.nrows()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn invert_identity() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let inv = invert(a.view()).unwrap();
        assert!((inv[[0, 0]] - 1.0).abs() < 1e-12);
        assert!(inv[[0, 1]].abs() < 1e-12);
    }

    #[test]
    fn softmax_matches_column_wise_scalar() {
        let dist = array![
            [0.1, 0.4, 1.2, 0.0],
            [0.8, 0.2, 0.3, 2.0],
            [1.5, 0.7, 0.05, 0.9],
        ];
        let sigma = array![0.1, 0.2, 0.15];
        let got = softmax_neg_dist(dist.view(), sigma.view());
        for j in 0..dist.ncols() {
            let mut max_logit = f64::NEG_INFINITY;
            let mut logits = vec![0.0f64; dist.nrows()];
            for i in 0..dist.nrows() {
                logits[i] = -dist[[i, j]] / sigma[i];
                if logits[i] > max_logit {
                    max_logit = logits[i];
                }
            }
            let mut sum = 0.0;
            for i in 0..dist.nrows() {
                sum += (logits[i] - max_logit).exp();
            }
            for i in 0..dist.nrows() {
                let want = (logits[i] - max_logit).exp() / sum;
                let diff = (got[[i, j]] - want).abs();
                assert!(diff < 1e-12, "softmax [{i},{j}] |Δ|={diff}");
            }
        }
    }

    #[test]
    fn invert_2x2() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = invert(a.view()).unwrap();
        let prod = a.dot(&inv);
        assert!((prod[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((prod[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(prod[[0, 1]].abs() < 1e-10);
    }
}
