use crate::error::{HarmonyError, Result};
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
    for j in 0..n {
        let mut max_logit = f64::NEG_INFINITY;
        for i in 0..k {
            let logit = -dist[[i, j]] / sigma[i];
            if logit > max_logit {
                max_logit = logit;
            }
        }
        let mut sum = 0.0;
        for i in 0..k {
            let v = (-dist[[i, j]] / sigma[i] - max_logit).exp();
            r[[i, j]] = v;
            sum += v;
        }
        if sum > 0.0 {
            for i in 0..k {
                r[[i, j]] /= sum;
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
    fn invert_2x2() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = invert(a.view()).unwrap();
        let prod = a.dot(&inv);
        assert!((prod[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((prod[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(prod[[0, 1]].abs() < 1e-10);
    }
}
