use crate::design::Design;
use crate::linalg::{cosine_distance, l2_normalize_columns, softmax_neg_dist};
use crate::r_rng::{ShuffleRng, UnitUniform};
use crate::vectormath::ln_nonpositive_zero;
use ndarray::{Array1, Array2, ArrayView2};

/// Harmony `kmeans_centers`: their k-means++ (`initialize_centroids`) plus
/// Euclidean Lloyd steps, then one L2 normalise. Matches `utils.cpp` in
/// immunogenomics/harmony 2.0.5 — not spherical k-means and not standard
/// min-distance k-means++. Arithmetic stays in `float` like Armadillo's
/// `MATTYPE = float` build. Empty-cluster recovery matches Armadillo's
/// `km_iterate` (steal last sample from a populous cluster, else `randi`).
pub fn kmeans_cosine(
    z: ArrayView2<'_, f64>,
    k: usize,
    rng: &mut impl UnitUniform,
    iters: usize,
) -> Array2<f64> {
    let n = z.ncols();
    let k = k.min(n).max(1);
    let z32 = z.mapv(|v| v as f32);
    let mut y32 = initialize_centroids(z32.view(), k, rng);
    for _ in 0..iters {
        y32 = euclidean_lloyd_step(z32.view(), y32.view(), rng);
    }
    l2_normalize_columns(y32.mapv(f64::from).view())
}

/// Harmony `initialize_centroids` (`src/utils.cpp`).
///
/// Distances at step `i` are to `Y.col(i)` only (not the min over chosen
/// centres). The last cell is excluded from the initial `floor(U*(N-1))` draw.
/// Draws use `(float) unif_rand()` like `arma::fill::randu` on a float build.
fn initialize_centroids(
    z: ArrayView2<'_, f32>,
    k: usize,
    rng: &mut impl UnitUniform,
) -> Array2<f32> {
    let d = z.nrows();
    let n = z.ncols();
    let mut y = Array2::<f32>::zeros((d, k));
    let n_init = n.saturating_sub(1).max(1);
    let n_init_f = n_init as f32;
    for c in 0..k {
        let u = rng.unit() as f32;
        let idx = ((u * n_init_f).floor() as usize).min(n - 1);
        y.column_mut(c).assign(&z.column(idx));
    }

    let mut taken = vec![false; n];
    for c in 0..k {
        let mut best_idx = 0usize;
        let mut best_score = f32::INFINITY;
        let mut scores = vec![f32::INFINITY; n];
        for j in 0..n {
            let mut sim = 0.0f32;
            for i in 0..d {
                sim += y[[i, c]] * z[[i, j]];
            }
            let dist = (2.0 * (1.0 - sim)).abs();
            let u = rng.unit() as f32;
            let score = if dist == 0.0 {
                f32::INFINITY
            } else {
                -u.ln() / dist
            };
            scores[j] = score;
            if score < best_score {
                best_score = score;
                best_idx = j;
            }
        }
        while taken[best_idx] {
            scores[best_idx] = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            best_idx = 0;
            best_score = f32::INFINITY;
            for j in 0..n {
                if scores[j] < best_score {
                    best_score = scores[j];
                    best_idx = j;
                }
            }
        }
        taken[best_idx] = true;
        y.column_mut(c).assign(&z.column(best_idx));
    }
    y
}

/// One `arma::kmeans(..., keep_existing, max_iter=1)` Euclidean Lloyd step in
/// `float`, including Armadillo's dead-mean recovery from `gmm_diag::km_iterate`.
fn euclidean_lloyd_step(
    z: ArrayView2<'_, f32>,
    y: ArrayView2<'_, f32>,
    rng: &mut impl UnitUniform,
) -> Array2<f32> {
    let d = z.nrows();
    let n = z.ncols();
    let k = y.ncols();
    let mut y_norm2 = vec![0.0f32; k];
    for c in 0..k {
        for i in 0..d {
            y_norm2[c] += y[[i, c]] * y[[i, c]];
        }
    }
    let mut counts = vec![0usize; k];
    let mut last_indx = vec![0usize; k];
    let mut next = Array2::<f32>::zeros((d, k));
    for j in 0..n {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += y[[i, c]] * z[[i, j]];
            }
            // ||z||^2 is constant across centroids for a fixed cell, so it
            // cancels in the argmin (same as Armadillo's full Euclidean).
            let dval = y_norm2[c] - 2.0 * dot;
            if dval < best_d {
                best_d = dval;
                best = c;
            }
        }
        counts[best] += 1;
        last_indx[best] = j;
        for i in 0..d {
            next[[i, best]] += z[[i, j]];
        }
    }
    for c in 0..k {
        if counts[c] >= 1 {
            let heft = counts[c] as f32;
            for i in 0..d {
                next[[i, c]] /= heft;
            }
        } else {
            // Armadillo zeros dead means before the recovery pass.
            for i in 0..d {
                next[[i, c]] = 0.0;
            }
        }
    }

    // `live_gs = sort(find(hefts >= 2), "descend")` — sorts cluster *ids*
    // descending, not hefts.
    let mut live_gs: Vec<usize> = (0..k).filter(|&c| counts[c] >= 2).collect();
    live_gs.sort_by(|a, b| b.cmp(a));
    let dead_gs: Vec<usize> = (0..k).filter(|&c| counts[c] == 0).collect();
    let mut live_gs_count = 0usize;
    for &dead_g in &dead_gs {
        let proposed = if live_gs_count < live_gs.len() {
            let live_g = live_gs[live_gs_count];
            live_gs_count += 1;
            last_indx[live_g]
        } else {
            // Last resort: `randi(0, N-1)` via R's `unif_rand` stream.
            rng.index(n)
        };
        next.column_mut(dead_g).assign(&z.column(proposed.min(n - 1)));
    }
    next
}

pub fn init_r(y: ArrayView2<'_, f64>, z: ArrayView2<'_, f64>, sigma: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let dist = cosine_distance(y, z);
    let r = softmax_neg_dist(dist.view(), sigma.view());
    (r, dist)
}

pub fn observed_expected(
    r: ArrayView2<'_, f64>,
    phi: ArrayView2<'_, f64>,
    pr_b: &[f64],
) -> (Array2<f64>, Array2<f64>) {
    // Harmony: `O = R * Φᵀ`, `E = sum(R,1) * Pr_bᵀ` in float.
    let r32 = r.mapv(|v| v as f32);
    let phi32 = phi.mapv(|v| v as f32);
    let o32 = r32.dot(&phi32.t());
    let k = r32.nrows();
    let b = phi32.nrows();
    let mut e32 = Array2::<f32>::zeros((k, b));
    for i in 0..k {
        let rk = r32.row(i).sum();
        for j in 0..b {
            e32[[i, j]] = rk * (pr_b[j] as f32);
        }
    }
    (o32.mapv(f64::from), e32.mapv(f64::from))
}

/// Online block update of soft assignments with the Harmony diversity penalty.
///
/// Mirrors Harmony `update_R`: Armadillo argsort-shuffle of columns, then online
/// blocks that drop `R` from `O`/`E`, recompute `R = softmax(-dist/σ) ⊙ diversity`,
/// and put the cells back — all in `float`. `r_base` is the distance-only softmax
/// (Harmony re-exps from `dist` each block; with fixed centroids that is the same
/// table). The diversity term `((2E+1)/(O+E+1))^θ` is a `K × B` table.
pub fn update_r(
    r: &mut Array2<f64>,
    r_base: ArrayView2<'_, f64>,
    design: &Design,
    theta: &[f64],
    o: &mut Array2<f64>,
    e: &mut Array2<f64>,
    block_size: f64,
    rng: &mut impl ShuffleRng,
) {
    let n = r.ncols();
    let k = r.nrows();
    let b = design.n_levels;
    let n_cov = design.n_covariates;
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle_cells(&mut order);

    // Harmony: `my_ceil(1.0 / block_size)` (float) and `unsigned(N * block_size)`.
    let block_size_f = block_size as f32;
    let n_blocks = my_ceil_f32(1.0 / block_size_f).max(1) as usize;
    let cells_per_block = ((n as f32 * block_size_f) as usize).max(1);
    let mut penalty = Array2::<f32>::zeros((k, b));
    let mut col = vec![0.0f32; k];
    let mut cluster_sum = vec![0.0f32; k];
    let pr_b32: Vec<f32> = design.pr_b.iter().map(|&p| p as f32).collect();
    let theta32: Vec<f32> = theta.iter().map(|&t| t as f32).collect();

    // Physical column shuffle like `R = R.cols(update_order)`.
    let mut r32 = Array2::<f32>::zeros((k, n));
    let mut base32 = Array2::<f32>::zeros((k, n));
    let mut batch_of = vec![0usize; n]; // primary covariate level in shuffled order
    // Multi-covariate: store every covariate's level per shuffled column.
    let mut levels_shuf = vec![vec![0usize; n]; n_cov];
    for (t, &j) in order.iter().enumerate() {
        for i in 0..k {
            r32[[i, t]] = r[[i, j]] as f32;
            base32[[i, t]] = r_base[[i, j]] as f32;
        }
        for ci in 0..n_cov {
            levels_shuf[ci][t] = design.global_level(ci, j);
        }
        batch_of[t] = design.global_level(0, j);
    }
    let mut o32 = o.mapv(|v| v as f32);
    let mut e32 = e.mapv(|v| v as f32);
    let _ = batch_of;

    for block in 0..n_blocks {
        let start = block * cells_per_block;
        if start >= n {
            break;
        }
        let end = if block + 1 == n_blocks {
            n
        } else {
            ((block + 1) * cells_per_block).min(n)
        };

        cluster_sum.fill(0.0);
        for t in start..end {
            for i in 0..k {
                let rij = r32[[i, t]];
                cluster_sum[i] += rij;
                for ci in 0..n_cov {
                    o32[[i, levels_shuf[ci][t]]] -= rij;
                }
            }
        }
        for i in 0..k {
            let s = cluster_sum[i];
            for bb in 0..b {
                e32[[i, bb]] -= s * pr_b32[bb];
            }
        }

        for i in 0..k {
            for bb in 0..b {
                let num = 2.0 * e32[[i, bb]] + 1.0;
                let den = o32[[i, bb]] + e32[[i, bb]] + 1.0;
                penalty[[i, bb]] = (num / den).powf(theta32[bb]);
            }
        }

        cluster_sum.fill(0.0);
        for t in start..end {
            let mut sum = 0.0f32;
            for i in 0..k {
                let mut p = base32[[i, t]];
                for ci in 0..n_cov {
                    p *= penalty[[i, levels_shuf[ci][t]]];
                }
                col[i] = p;
                sum += p;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for i in 0..k {
                    col[i] *= inv;
                }
            }
            for i in 0..k {
                let rij = col[i];
                r32[[i, t]] = rij;
                cluster_sum[i] += rij;
                for ci in 0..n_cov {
                    o32[[i, levels_shuf[ci][t]]] += rij;
                }
            }
        }
        for i in 0..k {
            let s = cluster_sum[i];
            for bb in 0..b {
                e32[[i, bb]] += s * pr_b32[bb];
            }
        }
    }

    // Unshuffle: `R = R.cols(reverse_index)`.
    for (t, &j) in order.iter().enumerate() {
        for i in 0..k {
            r[[i, j]] = f64::from(r32[[i, t]]);
        }
    }
    *o = o32.mapv(f64::from);
    *e = e32.mapv(f64::from);
}

/// Harmony `my_ceil` on a `float` (`utils.cpp`).
fn my_ceil_f32(num: f32) -> i32 {
    let inum = num as i32;
    if num == inum as f32 {
        inum
    } else {
        inum + 1
    }
}

pub fn objective(
    r: ArrayView2<'_, f64>,
    dist: ArrayView2<'_, f64>,
    o: ArrayView2<'_, f64>,
    e: ArrayView2<'_, f64>,
    design: &Design,
    theta: &[f64],
    sigma: &Array1<f64>,
) -> f64 {
    let n = r.ncols();
    let k = r.nrows();
    let b = design.n_levels;
    let n_cov = design.n_covariates;
    let norm = 2000.0 / n as f64;

    // `θ · log((O+E+1)/(2E+1))` depends only on cluster and batch. Harmony
    // builds this K×B table once; looking it up per cell avoids a `ln` and a
    // dense scan of Φ for every assignment.
    let mut log_table = Array2::<f64>::zeros((k, b));
    for i in 0..k {
        for bb in 0..b {
            let ratio = (o[[i, bb]] + e[[i, bb]] + 1.0) / (2.0 * e[[i, bb]] + 1.0);
            log_table[[i, bb]] = theta[bb] * ratio.ln();
        }
    }

    let mut kmeans_error = 0.0;
    let mut entropy = 0.0;
    let mut cross = 0.0;
    let mut ln_row = vec![0.0f64; n];

    for i in 0..k {
        let sigma_i = sigma[i];
        let rrow = r.row(i);
        let drow = dist.row(i);
        let rslice = rrow.as_slice().expect("objective R row is contiguous");
        ln_nonpositive_zero(&mut ln_row, rslice);
        for j in 0..n {
            let rij = rslice[j];
            kmeans_error += rij * drow[j];
            entropy += rij * ln_row[j] * sigma_i;
            let mut log_pen = 0.0;
            for ci in 0..n_cov {
                log_pen += log_table[[i, design.global_level(ci, j)]];
            }
            cross += rij * sigma_i * log_pen;
        }
    }

    (kmeans_error + entropy + cross) * norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::encode_labels;
    use crate::linalg::column_l1_normalize;
    use ndarray::{Array2, ArrayView2};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Previous per-cell `exp` / `pow` loop. Kept to prove the table-based
    /// update matches it on a fixed seed.
    fn reference_update_r(
        r: &mut Array2<f64>,
        dist: &Array2<f64>,
        design: &Design,
        theta: &[f64],
        sigma: &Array1<f64>,
        o: &mut Array2<f64>,
        e: &mut Array2<f64>,
        block_size: f64,
        rng: &mut impl ShuffleRng,
    ) {
        let n = r.ncols();
        let k = r.nrows();
        let b = design.n_levels;
        let n_cov = design.n_covariates;
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle_cells(&mut order);

        let n_blocks = ((1.0 / block_size).ceil() as usize).max(1);
        let cells_per_block = ((n as f64 * block_size) as usize).max(1);

        for block in 0..n_blocks {
            let start = block * cells_per_block;
            if start >= n {
                break;
            }
            let end = if block + 1 == n_blocks {
                n
            } else {
                ((block + 1) * cells_per_block).min(n)
            };
            let cells = &order[start..end];

            for &j in cells {
                for i in 0..k {
                    let rij = r[[i, j]];
                    for bb in 0..b {
                        e[[i, bb]] -= rij * design.pr_b[bb];
                    }
                    for ci in 0..n_cov {
                        let bb = design.global_level(ci, j);
                        o[[i, bb]] -= rij;
                    }
                }
            }

            let mut block_r = Array2::<f64>::zeros((k, cells.len()));
            for (t, &j) in cells.iter().enumerate() {
                let mut max_logit = f64::NEG_INFINITY;
                for i in 0..k {
                    max_logit = max_logit.max(-dist[[i, j]] / sigma[i]);
                }
                let mut sum = 0.0;
                for i in 0..k {
                    let v = (-dist[[i, j]] / sigma[i] - max_logit).exp();
                    block_r[[i, t]] = v;
                    sum += v;
                }
                if sum > 0.0 {
                    for i in 0..k {
                        block_r[[i, t]] /= sum;
                    }
                }
            }

            for (t, &j) in cells.iter().enumerate() {
                for i in 0..k {
                    let mut penalty = 1.0;
                    for ci in 0..n_cov {
                        let bb = design.global_level(ci, j);
                        let num = 2.0 * e[[i, bb]] + 1.0;
                        let den = o[[i, bb]] + e[[i, bb]] + 1.0;
                        penalty *= (num / den).powf(theta[bb]);
                    }
                    block_r[[i, t]] *= penalty;
                }
            }
            block_r = column_l1_normalize(block_r);

            for (t, &j) in cells.iter().enumerate() {
                for i in 0..k {
                    r[[i, j]] = block_r[[i, t]];
                    let rij = r[[i, j]];
                    for bb in 0..b {
                        e[[i, bb]] = (e[[i, bb]] + rij * design.pr_b[bb]).max(0.0);
                    }
                    for ci in 0..n_cov {
                        let bb = design.global_level(ci, j);
                        o[[i, bb]] = (o[[i, bb]] + rij).max(0.0);
                    }
                }
            }
        }
    }

    fn tiny_state(n_cov: usize) -> (Array2<f64>, Array2<f64>, Array1<f64>, Design, Vec<f64>) {
        let n = 40usize;
        let k = 6usize;
        let d = 4usize;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut z = Array2::<f64>::zeros((d, n));
        for j in 0..n {
            for i in 0..d {
                z[[i, j]] = rng.gen::<f64>() - 0.5;
            }
        }
        z = l2_normalize_columns(z.view());
        let y = kmeans_cosine(z.view(), k, &mut rng, 4);
        let sigma = Array1::from_elem(k, 0.1);
        let (r, dist) = init_r(y.view(), z.view(), &sigma);
        let batch: Vec<String> = (0..n)
            .map(|j| if j % 2 == 0 { "A".into() } else { "B".into() })
            .collect();
        let design = if n_cov == 1 {
            encode_labels(&batch).unwrap()
        } else {
            let proto: Vec<String> = (0..n)
                .map(|j| if j % 3 == 0 { "v2".into() } else { "v3".into() })
                .collect();
            crate::design::encode_covariates(&[batch, proto]).unwrap()
        };
        let theta = vec![2.0; design.n_levels];
        let _ = dist;
        (r, z, sigma, design, theta)
    }

    #[test]
    fn update_r_matches_per_cell_reference() {
        for n_cov in [1usize, 2] {
            let (r0, z, sigma, design, theta) = tiny_state(n_cov);
            let y = kmeans_cosine(z.view(), r0.nrows(), &mut ChaCha8Rng::seed_from_u64(9), 2);
            let (r, dist) = init_r(y.view(), z.view(), &sigma);
            let (o, e) = observed_expected(r.view(), design.phi.view(), &design.pr_b);
            let r_base = softmax_neg_dist(dist.view(), sigma.view());

            let mut r_fast = r.clone();
            let mut o_fast = o.clone();
            let mut e_fast = e.clone();
            let mut r_slow = r.clone();
            let mut o_slow = o.clone();
            let mut e_slow = e.clone();

            update_r(
                &mut r_fast,
                r_base.view(),
                &design,
                &theta,
                &mut o_fast,
                &mut e_fast,
                0.2,
                &mut ChaCha8Rng::seed_from_u64(21),
            );
            reference_update_r(
                &mut r_slow,
                &dist,
                &design,
                &theta,
                &sigma,
                &mut o_slow,
                &mut e_slow,
                0.2,
                &mut ChaCha8Rng::seed_from_u64(21),
            );

            for i in 0..r.nrows() {
                for j in 0..r.ncols() {
                    let diff = (r_fast[[i, j]] - r_slow[[i, j]]).abs();
                    assert!(
                        diff < 1e-6,
                        "R mismatch at [{i},{j}] n_cov={n_cov}: {diff}"
                    );
                }
            }
            for i in 0..o.nrows() {
                for bb in 0..o.ncols() {
                    assert!((o_fast[[i, bb]] - o_slow[[i, bb]]).abs() < 1e-6);
                    assert!((e_fast[[i, bb]] - e_slow[[i, bb]]).abs() < 1e-6);
                }
            }
            for j in 0..r.ncols() {
                let col: f64 = (0..r.nrows()).map(|i| r_fast[[i, j]]).sum();
                // Float L1 renormalise (Harmony) leaves ~1 ulp when widened to f64.
                assert!((col - 1.0).abs() < 1e-6, "column {j} summed to {col}");
            }
        }
    }

    #[test]
    fn kmeans_cosine_is_deterministic_for_a_fixed_rng() {
        let d = 8usize;
        let n = 120usize;
        let k = 25usize;
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let mut z = Array2::<f64>::zeros((d, n));
        for j in 0..n {
            for i in 0..d {
                z[[i, j]] = rng.gen::<f64>() - 0.5;
            }
        }
        z = l2_normalize_columns(z.view());

        let a = kmeans_cosine(z.view(), k, &mut ChaCha8Rng::seed_from_u64(11), 6);
        let b = kmeans_cosine(z.view(), k, &mut ChaCha8Rng::seed_from_u64(11), 6);
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), b.ncols());
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let diff = (a[[i, j]] - b[[i, j]]).abs();
                assert!(diff < 1e-12, "centroid mismatch at [{i},{j}]: {diff}");
            }
        }
    }

    /// Previous objective: `ln` and a dense Φ scan per assignment.
    fn reference_objective(
        r: ArrayView2<'_, f64>,
        dist: ArrayView2<'_, f64>,
        o: ArrayView2<'_, f64>,
        e: ArrayView2<'_, f64>,
        phi: ArrayView2<'_, f64>,
        theta: &[f64],
        sigma: &Array1<f64>,
    ) -> f64 {
        let n = r.ncols() as f64;
        let k = r.nrows();
        let b = phi.nrows();
        let norm = 2000.0 / n;
        let mut kmeans_error = 0.0;
        let mut entropy = 0.0;
        let mut cross = 0.0;
        for i in 0..k {
            for j in 0..r.ncols() {
                let rij = r[[i, j]];
                kmeans_error += rij * dist[[i, j]];
                if rij > 0.0 {
                    entropy += rij * rij.ln() * sigma[i];
                }
                let mut log_pen = 0.0;
                for bb in 0..b {
                    if phi[[bb, j]] == 0.0 {
                        continue;
                    }
                    let ratio = (o[[i, bb]] + e[[i, bb]] + 1.0) / (2.0 * e[[i, bb]] + 1.0);
                    log_pen += theta[bb] * ratio.ln();
                }
                cross += rij * sigma[i] * log_pen;
            }
        }
        (kmeans_error + entropy + cross) * norm
    }

    #[test]
    fn objective_log_table_matches_dense_phi_scan() {
        for n_cov in [1usize, 2] {
            let (r, z, sigma, design, theta) = tiny_state(n_cov);
            let y = kmeans_cosine(z.view(), r.nrows(), &mut ChaCha8Rng::seed_from_u64(9), 2);
            let (r, dist) = init_r(y.view(), z.view(), &sigma);
            let (o, e) = observed_expected(r.view(), design.phi.view(), &design.pr_b);
            let fast = objective(
                r.view(),
                dist.view(),
                o.view(),
                e.view(),
                &design,
                &theta,
                &sigma,
            );
            let slow = reference_objective(
                r.view(),
                dist.view(),
                o.view(),
                e.view(),
                design.phi.view(),
                &theta,
                &sigma,
            );
            assert!(
                (fast - slow).abs() < 1e-9 * slow.abs().max(1.0),
                "n_cov={n_cov}: fast={fast} slow={slow}"
            );
        }
    }
}
