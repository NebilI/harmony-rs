use crate::error::{HarmonyError, Result};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, ShapeBuilder};

/// Column-major (Fortran) layout so each cell / cluster column is contiguous.
pub fn zeros_f32_f(nrows: usize, ncols: usize) -> Array2<f32> {
    Array2::<f32>::zeros((nrows, ncols).f())
}

/// Column-major `f64` workspace matching Armadillo's default storage.
pub fn zeros_f64_f(nrows: usize, ncols: usize) -> Array2<f64> {
    Array2::<f64>::zeros((nrows, ncols).f())
}

/// Two-accumulator `sum x²` matching Armadillo's small-n `vec_norm_2`.
pub fn accu_sq_two(xs: impl IntoIterator<Item = f32>) -> f32 {
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut toggle = false;
    for x in xs {
        if toggle {
            acc2 += x * x;
        } else {
            acc1 += x * x;
        }
        toggle = !toggle;
    }
    acc1 + acc2
}

/// Armadillo `distance<eT,1>`: paired `(A-B)²` with two accumulators.
pub fn euclid2_two_acc(a: impl Fn(usize) -> f32, b: impl Fn(usize) -> f32, n: usize) -> f32 {
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut i = 0usize;
    let mut j = 1usize;
    while j < n {
        let ti = a(i) - b(i);
        let tj = a(j) - b(j);
        acc1 += ti * ti;
        acc2 += tj * tj;
        i += 2;
        j += 2;
    }
    if i < n {
        let ti = a(i) - b(i);
        acc1 += ti * ti;
    }
    acc1 + acc2
}

pub fn l2_normalize_columns(z: ArrayView2<'_, f64>) -> Array2<f64> {
    // Harmony / Armadillo: two-accumulator ||x||₂ when d < 32 (typical PC count).
    let mut out = z.mapv(|v| v as f32);
    let d = out.nrows();
    for mut col in out.columns_mut() {
        let norm = if d < 32 {
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut i = 0usize;
            while i + 1 < d {
                acc1 += col[i] * col[i];
                acc2 += col[i + 1] * col[i + 1];
                i += 2;
            }
            if i < d {
                acc1 += col[i] * col[i];
            }
            (acc1 + acc2).sqrt()
        } else {
            col.dot(&col).sqrt()
        };
        if norm > 0.0 {
            col.mapv_inplace(|v| v / norm);
        }
    }
    out.mapv(|v| f64::from(v))
}

pub fn cosine_distance(y: ArrayView2<'_, f64>, z: ArrayView2<'_, f64>) -> Array2<f64> {
    // Y is d x K, Z is d x N. Dist = 2 (1 - Y^T Z), evaluated in float.
    let y32 = y.mapv(|v| v as f32);
    let z32 = z.mapv(|v| v as f32);
    let sim = y32.t().dot(&z32);
    sim.mapv(|s| f64::from(2.0 * (1.0 - s)))
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

/// Gauss–Jordan inverse in `float`, matching Harmony's single-precision ridge.
pub fn invert_f32(a: ArrayView2<'_, f32>) -> Result<Array2<f32>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(HarmonyError::msg("invert: matrix is not square"));
    }
    let mut m = Array2::<f32>::zeros((n, 2 * n));
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

/// Harmony's float arrowhead inverse (`moe_correct_ridge_cpp`).
pub fn invert_arrowhead_f32(phi_cov: ArrayView2<'_, f32>) -> Result<Array2<f32>> {
    let n = phi_cov.nrows();
    let mut ac = Array1::<f32>::zeros(n);
    ac[0] = 1.0;
    for i in 1..n {
        ac[i] = -phi_cov[[0, i]];
    }
    let b0 = phi_cov[[0, 0]];
    let mut b = Array1::<f32>::zeros(n);
    for i in 1..n {
        let d = phi_cov[[i, i]];
        if d.abs() < 1e-18 {
            return invert_f32(phi_cov);
        }
        b[i] = 1.0 / d;
    }
    let mut u = b0;
    for i in 1..n {
        u -= ac[i] * ac[i] * b[i];
    }
    if u.abs() < 1e-18 {
        return invert_f32(phi_cov);
    }
    let mut ac_b = Array1::<f32>::zeros(n);
    ac_b[0] = 1.0;
    for i in 1..n {
        ac_b[i] = ac[i] * b[i];
    }
    let mut inv = Array2::<f32>::zeros((n, n));
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
    if n == 0 || k == 0 {
        return Array2::<f64>::zeros((k, n));
    }

    // Harmony C++: `R = -dist; R.each_col() /= sigma; R = exp(R)` then
    // `R.each_row() /= sum(R, 0)`. Division, not multiply-by-reciprocal.
    // Cosine distances stay in [0, 4], so expf does not overflow at σ=0.1.
    let mut r32 = zeros_f32_f(k, n);
    for i in 0..k {
        let sig = sigma[i] as f32;
        let drow = dist.row(i);
        for j in 0..n {
            r32[[i, j]] = (-(drow[j] as f32) / sig).exp();
        }
    }
    for j in 0..n {
        // Armadillo `sum(R, 0)` on a column: two-accumulator over K.
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut i = 0usize;
        while i + 1 < k {
            acc1 += r32[[i, j]];
            acc2 += r32[[i + 1, j]];
            i += 2;
        }
        if i < k {
            acc1 += r32[[i, j]];
        }
        let s = acc1 + acc2;
        if s > 0.0 {
            for i in 0..k {
                r32[[i, j]] /= s;
            }
        }
    }
    r32.mapv(|v| f64::from(v))
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
            let mut sum = 0.0f32;
            let mut vals = vec![0.0f32; dist.nrows()];
            for i in 0..dist.nrows() {
                vals[i] = (-(dist[[i, j]] as f32) / (sigma[i] as f32)).exp();
                sum += vals[i];
            }
            for i in 0..dist.nrows() {
                let want = f64::from(vals[i] / sum);
                let diff = (got[[i, j]] - want).abs();
                assert!(diff < 1e-7, "softmax [{i},{j}] |Δ|={diff}");
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
