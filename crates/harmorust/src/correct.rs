use crate::design::Design;
use crate::error::Result;
use crate::linalg::{invert_arrowhead_f32, invert_f32, l2_normalize_columns};
use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Mixture-of-experts ridge correction from the original embeddings.
///
/// Each Harmony round starts from `Z_orig` (not the previous correction) and
/// subtracts the batch coefficients of a per-cluster ridge fit. Cluster fits
/// are independent and run in parallel. Right-hand sides and the correction
/// itself walk `Design::index` (cells of each batch) instead of testing every
/// cell against every kept level.
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
    let z32 = z_orig.mapv(|v| v as f32);
    let r32 = r.mapv(|v| v as f32);
    let o32 = o.mapv(|v| v as f32);
    let e32 = e.mapv(|v| v as f32);
    let k = r32.nrows();
    let b = design.n_levels;
    let lambda_owned = lambda.map(|l| l.to_vec());

    let fits: Result<Vec<Option<ClusterFit>>> = (0..k)
        .into_par_iter()
        .map(|cluster| {
            fit_cluster(
                z32.view(),
                r32.row(cluster),
                o32.row(cluster),
                e32.row(cluster),
                design,
                lambda_owned.as_deref(),
                alpha,
                batch_prop_cutoff,
                b,
            )
        })
        .collect();
    let fits = fits?;

    let mut z_corr = z32;
    let d = z_orig.nrows();
    let n = z_orig.ncols();
    let mut y32 = y.mapv(|v| v as f32);
    for (cluster, fit) in fits.iter().enumerate() {
        if let Some(fit) = fit {
            y32.column_mut(cluster).assign(&fit.w_coef.row(0));
        }
    }
    // Harmony applies clusters in order. Walk cells once and subtract every
    // cluster's term for that cell in the same cluster order (bit-identical
    // to a per-cluster pass: each cell only sees sequential `-=`).
    let mut acc = vec![0.0f32; d];
    for j in 0..n {
        for (cluster, fit) in fits.iter().enumerate() {
            let Some(fit) = fit else {
                continue;
            };
            let w = r32[[cluster, j]];
            if w == 0.0 {
                continue;
            }
            acc.fill(0.0);
            for (t, &bb) in fit.keep.iter().enumerate() {
                let mut belongs = false;
                for ci in 0..design.n_covariates {
                    if design.global_level(ci, j) == bb {
                        belongs = true;
                        break;
                    }
                }
                if !belongs {
                    continue;
                }
                let coef = fit.w_coef.row(t + 1);
                for i in 0..d {
                    acc[i] += coef[i] * w;
                }
            }
            for i in 0..d {
                z_corr[[i, j]] -= acc[i];
            }
        }
    }

    let y64 = l2_normalize_columns(y32.mapv(f64::from).view());
    *y = y64;
    Ok(z_corr.mapv(f64::from))
}

struct ClusterFit {
    keep: Vec<usize>,
    w_coef: Array2<f32>,
}

// Cluster fits stay independent; the fused apply above walks cells once.

fn fit_cluster(
    z_orig: ArrayView2<'_, f32>,
    rk: ArrayView1<'_, f32>,
    o_row: ArrayView1<'_, f32>,
    e_row: ArrayView1<'_, f32>,
    design: &Design,
    lambda: Option<&[f64]>,
    alpha: f64,
    batch_prop_cutoff: f64,
    b: usize,
) -> Result<Option<ClusterFit>> {
    let d = z_orig.nrows();
    let mut keep: Vec<usize> = Vec::new();
    let mut levels_kept = vec![0usize; design.n_covariates];
    let mut offset = 0usize;
    for (ci, &nlev) in design.levels_per_covariate.iter().enumerate() {
        for local in 0..nlev {
            let bb = offset + local;
            let avg = if design.batch_sizes[bb] > 0.0 {
                f64::from(o_row[bb]) / design.batch_sizes[bb]
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
        return Ok(None);
    }

    // Harmony builds Φ_cov / RHS / inv in `float` (OpenBLAS sgetrf or the
    // analytic arrowhead). Stay in f32 end-to-end so |Δ| is only residual
    // GEMM accumulation order, not a float↔f64 type gap.
    let lambda_vec: Vec<f32> = match lambda {
        Some(l) => l.iter().map(|&v| v as f32).collect(),
        None => {
            let mut v = vec![0.0f32];
            for bb in 0..b {
                v.push((alpha as f32) * e_row[bb]);
            }
            v
        }
    };

    let m = keep_final.len() + 1;
    let mut phi_cov = Array2::<f32>::zeros((m, m));
    let r_sum = rk.sum();
    phi_cov[[0, 0]] = r_sum + lambda_vec[0];
    for (t, &bb) in keep_final.iter().enumerate() {
        let ob = o_row[bb];
        phi_cov[[0, t + 1]] = ob;
        phi_cov[[t + 1, 0]] = ob;
        phi_cov[[t + 1, t + 1]] = ob + lambda_vec.get(bb + 1).copied().unwrap_or(1.0);
    }

    // Harmony uses the float arrowhead whenever there is one covariate,
    // including when a batch has been dropped (Φ_cov is still arrowhead).
    let inv_cov = if design.n_covariates == 1 {
        invert_arrowhead_f32(phi_cov.view()).or_else(|_| invert_f32(phi_cov.view()))?
    } else {
        invert_f32(phi_cov.view())?
    };

    // Match Harmony: scale Z columns by R_k, sum per batch (cell index order),
    // intercept = total sum (or sum of batch RHSs for a full one-hot partition).
    let mut rhs = Array2::<f32>::zeros((m, d));
    for (t, &bb) in keep_final.iter().enumerate() {
        for &j in &design.index[bb] {
            let w = rk[j];
            if w == 0.0 {
                continue;
            }
            for i in 0..d {
                rhs[[t + 1, i]] += w * z_orig[[i, j]];
            }
        }
    }
    if design.n_covariates == 1 && keep_final.len() == b {
        for t in 0..keep_final.len() {
            for i in 0..d {
                rhs[[0, i]] += rhs[[t + 1, i]];
            }
        }
    } else {
        for j in 0..rk.len() {
            let w = rk[j];
            if w == 0.0 {
                continue;
            }
            for i in 0..d {
                rhs[[0, i]] += w * z_orig[[i, j]];
            }
        }
    }

    Ok(Some(ClusterFit {
        keep: keep_final,
        w_coef: inv_cov.dot(&rhs),
    }))
}

pub fn default_lambda(n_levels: usize, lambda: f64) -> Vec<f64> {
    let mut v = vec![0.0];
    v.extend(std::iter::repeat(lambda).take(n_levels));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::{init_r, kmeans_cosine, observed_expected};
    use crate::design::encode_covariates;
    use crate::harmony::embeddings_from_rows;
    use crate::linalg::{invert, l2_normalize_columns};
    use crate::simulate::{simulate_atlas, SimulateSpec};
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn reference_moe(
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
        let k = r.nrows();
        let b = design.n_levels;
        let d = z_orig.nrows();
        let n = z_orig.ncols();
        let mut z_corr = z_orig.to_owned();
        for cluster in 0..k {
            let rk = r.row(cluster);
            let o_row = o.row(cluster);
            let e_row = e.row(cluster);
            let mut keep: Vec<usize> = Vec::new();
            let mut levels_kept = vec![0usize; design.n_covariates];
            let mut offset = 0usize;
            for (ci, &nlev) in design.levels_per_covariate.iter().enumerate() {
                for local in 0..nlev {
                    let bb = offset + local;
                    let avg = if design.batch_sizes[bb] > 0.0 {
                        o_row[bb] / design.batch_sizes[bb]
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
                        v.push(alpha * e_row[bb]);
                    }
                    v
                }
            };
            let m = keep_final.len() + 1;
            let mut phi_cov = Array2::<f64>::zeros((m, m));
            phi_cov[[0, 0]] = rk.sum() + lambda_vec[0];
            for (t, &bb) in keep_final.iter().enumerate() {
                let ob = o_row[bb];
                phi_cov[[0, t + 1]] = ob;
                phi_cov[[t + 1, 0]] = ob;
                phi_cov[[t + 1, t + 1]] = ob + lambda_vec.get(bb + 1).copied().unwrap_or(1.0);
            }
            let inv_cov = invert(phi_cov.view())?;
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
                    let mut belongs = false;
                    for ci in 0..design.n_covariates {
                        if design.global_level(ci, j) == bb {
                            belongs = true;
                            break;
                        }
                    }
                    if belongs {
                        for i in 0..d {
                            rhs[[t + 1, i]] += w * z_orig[[i, j]];
                        }
                    }
                }
            }
            let w_coef = inv_cov.dot(&rhs);
            for i in 0..d {
                y[[i, cluster]] = w_coef[[0, i]];
            }
            for j in 0..n {
                let w = rk[j];
                if w == 0.0 {
                    continue;
                }
                for (t, &bb) in keep_final.iter().enumerate() {
                    let mut belongs = false;
                    for ci in 0..design.n_covariates {
                        if design.global_level(ci, j) == bb {
                            belongs = true;
                            break;
                        }
                    }
                    if !belongs {
                        continue;
                    }
                    for i in 0..d {
                        z_corr[[i, j]] -= w_coef[[t + 1, i]] * w;
                    }
                }
            }
        }
        *y = l2_normalize_columns(y.view());
        Ok(z_corr)
    }

    fn ridge_inputs(n_cov: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Design) {
        let spec = SimulateSpec {
            n_cells: 80,
            n_pcs: 6,
            n_batches: 3,
            n_types: 4,
            batch_strength: 1.8,
            seed: 5,
        };
        let data = simulate_atlas(&spec);
        let z_cells = embeddings_from_rows(&data.embeddings).unwrap();
        let z = l2_normalize_columns(z_cells.t());
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let y = kmeans_cosine(z.view(), 5, &mut rng, 4);
        let sigma = Array1::from_elem(5, 0.1);
        let (r, _) = init_r(y.view(), z.view(), &sigma);
        let donor = data.batch;
        let design = if n_cov == 1 {
            crate::design::encode_labels(&donor).unwrap()
        } else {
            let proto: Vec<String> = (0..data.n_cells)
                .map(|i| if i % 2 == 0 { "v2".into() } else { "v3".into() })
                .collect();
            encode_covariates(&[donor, proto]).unwrap()
        };
        let (o, e) = observed_expected(r.view(), design.phi.view(), &design.pr_b);
        (z, r, o, e, design)
    }

    #[test]
    fn ridge_index_lists_match_per_cell_membership_scan() {
        for n_cov in [1usize, 2] {
            let (z, r, o, e, design) = ridge_inputs(n_cov);
            let lambda = crate::correct::default_lambda(design.n_levels, 1.0);
            let mut y_fast = Array2::<f64>::zeros((z.nrows(), r.nrows()));
            let mut y_slow = y_fast.clone();
            let fast = moe_correct_ridge(
                z.view(),
                r.view(),
                o.view(),
                e.view(),
                &design,
                Some(&lambda),
                0.2,
                1e-5,
                &mut y_fast,
            )
            .unwrap();
            let slow = reference_moe(
                z.view(),
                r.view(),
                o.view(),
                e.view(),
                &design,
                Some(&lambda),
                0.2,
                1e-5,
                &mut y_slow,
            )
            .unwrap();
            for i in 0..fast.nrows() {
                for j in 0..fast.ncols() {
                    let diff = (fast[[i, j]] - slow[[i, j]]).abs();
                    assert!(
                        diff < 2e-6,
                        "Z mismatch n_cov={n_cov} at [{i},{j}]: {diff}"
                    );
                }
            }
            for i in 0..y_fast.nrows() {
                for j in 0..y_fast.ncols() {
                    let diff = (y_fast[[i, j]] - y_slow[[i, j]]).abs();
                    assert!(diff < 1e-6, "Y mismatch n_cov={n_cov} at [{i},{j}]: {diff}");
                }
            }
        }
    }
}
