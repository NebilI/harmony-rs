use crate::design::Design;
use crate::error::Result;
use crate::linalg::{invert, invert_arrowhead, l2_normalize_columns};
use ndarray::{Array2, ArrayView2};

/// Mixture-of-experts ridge correction from the original embeddings.
///
/// Each Harmony round starts from `Z_orig` (not the previous correction) and
/// subtracts the batch coefficients of a per-cluster ridge fit.
pub fn moe_correct_ridge(
    z_orig: ArrayView2<'_, f64>,
    r: ArrayView2<'_, f64>,
    o: ArrayView2<'_, f64>,
    e: ArrayView2<'_, f64>,
    design: &Design,
    lambda: Option<&[f64]>,
    alpha: f64,
    batch_prop_cutoff: f64,
    y: &mut Array2<f64>,
) -> Result<Array2<f64>> {
    let d = z_orig.nrows();
    let n = z_orig.ncols();
    let k = r.nrows();
    let b = design.n_levels;
    let mut z_corr = z_orig.to_owned();

    for cluster in 0..k {
        let rk = r.row(cluster);
        let mut keep: Vec<usize> = Vec::new();
        let mut levels_kept = vec![0usize; design.n_covariates];
        let mut offset = 0usize;
        for (ci, &nlev) in design.levels_per_covariate.iter().enumerate() {
            for local in 0..nlev {
                let bb = offset + local;
                let avg = if design.batch_sizes[bb] > 0.0 {
                    o[[cluster, bb]] / design.batch_sizes[bb]
                } else {
                    0.0
                };
                if avg > batch_prop_cutoff {
                    levels_kept[ci] += 1;
                    keep.push(bb);
                }
            }
            offset += nlev;
        }

        // Drop a covariate entirely when only one of its levels qualifies.
        let mut keep_final = Vec::new();
        offset = 0;
        for (ci, &nlev) in design.levels_per_covariate.iter().enumerate() {
            if levels_kept[ci] > 1 {
                for local in 0..nlev {
                    let bb = offset + local;
                    if keep.contains(&bb) {
                        keep_final.push(bb);
                    }
                }
            }
            offset += nlev;
        }
        if keep_final.is_empty() {
            continue;
        }

        let lambda_vec = match lambda {
            Some(l) => l.to_vec(),
            None => {
                let mut v = vec![0.0];
                for bb in 0..b {
                    v.push(alpha * e[[cluster, bb]]);
                }
                v
            }
        };

        let m = keep_final.len() + 1;
        let mut phi_cov = Array2::<f64>::zeros((m, m));
        let r_sum: f64 = rk.sum();
        phi_cov[[0, 0]] = r_sum + lambda_vec[0];
        for (t, &bb) in keep_final.iter().enumerate() {
            let ob = o[[cluster, bb]];
            phi_cov[[0, t + 1]] = ob;
            phi_cov[[t + 1, 0]] = ob;
            phi_cov[[t + 1, t + 1]] = ob + lambda_vec.get(bb + 1).copied().unwrap_or(1.0);
        }

        let inv_cov = if design.n_covariates == 1 && keep_final.len() == b {
            invert_arrowhead(phi_cov.view()).or_else(|_| invert(phi_cov.view()))?
        } else {
            invert(phi_cov.view())?
        };

        // rhs is m x d: weighted sums of original PCs
        let mut rhs = Array2::<f64>::zeros((m, d));
        for j in 0..n {
            let w = rk[j];
            if w == 0.0 {
                continue;
            }
            for i in 0..d {
                rhs[[0, i]] += w * z_orig[[i, j]];
            }
            for (t, &bb) in keep_final.iter().enumerate() {
                if design.phi[[bb, j]] != 0.0 {
                    for i in 0..d {
                        rhs[[t + 1, i]] += w * z_orig[[i, j]];
                    }
                }
            }
        }

        let w_coef = inv_cov.dot(&rhs); // m x d
        for i in 0..d {
            y[[i, cluster]] = w_coef[[0, i]];
        }

        // Do not remove the intercept. Subtract batch coefficients only.
        for j in 0..n {
            let w = rk[j];
            if w == 0.0 {
                continue;
            }
            for (t, &bb) in keep_final.iter().enumerate() {
                if design.phi[[bb, j]] != 0.0 {
                    for i in 0..d {
                        z_corr[[i, j]] -= w_coef[[t + 1, i]] * w;
                    }
                }
            }
        }
    }

    *y = l2_normalize_columns(y.view());
    Ok(z_corr)
}

pub fn default_lambda(n_levels: usize, lambda: f64) -> Vec<f64> {
    let mut v = vec![0.0];
    v.extend(std::iter::repeat(lambda).take(n_levels));
    v
}
