use crate::design::Design;
use crate::linalg::{cosine_distance, l2_normalize_columns, softmax_neg_dist};
use crate::vectormath::ln_nonpositive_zero;
use ndarray::{Array1, Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;

/// k-means++ then a few Lloyd iterations on L2-normalised embeddings.
pub fn kmeans_cosine(
    z: ArrayView2<'_, f64>,
    k: usize,
    rng: &mut impl Rng,
    iters: usize,
) -> Array2<f64> {
    let d = z.nrows();
    let n = z.ncols();
    let k = k.min(n).max(1);
    let mut y = Array2::<f64>::zeros((d, k));

    let first = rng.gen_range(0..n);
    y.column_mut(0).assign(&z.column(first));
    let mut chosen = vec![false; n];
    chosen[first] = true;
    // Running nearest-centroid cosine distance. After each new seed we only
    // compare against that seed, not the whole chosen set.
    let mut min_dist = vec![f64::INFINITY; n];

    for c in 1..k {
        let prev = c - 1;
        for j in 0..n {
            let mut sim = 0.0;
            for i in 0..d {
                sim += y[[i, prev]] * z[[i, j]];
            }
            let dist = (2.0 * (1.0 - sim)).abs().max(1e-12);
            if dist < min_dist[j] {
                min_dist[j] = dist;
            }
        }
        // Harmony's weighted sample: pick argmin of -log(U) / dist
        let mut best_idx = 0usize;
        let mut best_score = f64::INFINITY;
        for j in 0..n {
            if chosen[j] {
                continue;
            }
            let u: f64 = rng.gen::<f64>().clamp(1e-12, 1.0);
            let score = -u.ln() / min_dist[j];
            if score < best_score {
                best_score = score;
                best_idx = j;
            }
        }
        chosen[best_idx] = true;
        y.column_mut(c).assign(&z.column(best_idx));
    }

    y = l2_normalize_columns(y.view());

    for _ in 0..iters {
        let dist = cosine_distance(y.view(), z);
        let mut counts = vec![0.0f64; k];
        let mut next = Array2::<f64>::zeros((d, k));
        for j in 0..n {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let dval = dist[[c, j]];
                if dval < best_d {
                    best_d = dval;
                    best = c;
                }
            }
            counts[best] += 1.0;
            for i in 0..d {
                next[[i, best]] += z[[i, j]];
            }
        }
        for c in 0..k {
            if counts[c] > 0.0 {
                for i in 0..d {
                    next[[i, c]] /= counts[c];
                }
            } else {
                next.column_mut(c).assign(&y.column(c));
            }
        }
        y = l2_normalize_columns(next.view());
    }

    y
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
    let o = r.dot(&phi.t());
    let k = r.nrows();
    let b = phi.nrows();
    let mut e = Array2::<f64>::zeros((k, b));
    for i in 0..k {
        let rk = r.row(i).sum();
        for j in 0..b {
            e[[i, j]] = rk * pr_b[j];
        }
    }
    (o, e)
}

/// Online block update of soft assignments with the Harmony diversity penalty.
///
/// `r_base` is the distance-only softmax `softmax(-dist / σ)`. Centroids (and
/// therefore distances) are fixed for a clustering round, so the caller computes
/// it once and we reuse it for every inner pass and every block. The diversity
/// term `((2E+1)/(O+E+1))^θ` lives on the cluster × batch table, so `pow` runs
/// `K · B` times per block instead of once per cell.
pub fn update_r(
    r: &mut Array2<f64>,
    r_base: ArrayView2<'_, f64>,
    design: &Design,
    theta: &[f64],
    o: &mut Array2<f64>,
    e: &mut Array2<f64>,
    block_size: f64,
    rng: &mut impl Rng,
) {
    let n = r.ncols();
    let k = r.nrows();
    let b = design.n_levels;
    let n_cov = design.n_covariates;
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);

    let n_blocks = ((1.0 / block_size).ceil() as usize).max(1);
    let cells_per_block = ((n as f64 * block_size) as usize).max(1);
    let mut penalty = Array2::<f64>::zeros((k, b));
    let mut col = vec![0.0f64; k];
    let mut cluster_sum = vec![0.0f64; k];

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

        cluster_sum.fill(0.0);
        for &j in cells {
            for i in 0..k {
                let rij = r[[i, j]];
                cluster_sum[i] += rij;
                for ci in 0..n_cov {
                    o[[i, design.global_level(ci, j)]] -= rij;
                }
            }
        }
        for i in 0..k {
            let s = cluster_sum[i];
            for bb in 0..b {
                e[[i, bb]] -= s * design.pr_b[bb];
            }
        }

        for i in 0..k {
            for bb in 0..b {
                let num = 2.0 * e[[i, bb]] + 1.0;
                let den = o[[i, bb]] + e[[i, bb]] + 1.0;
                penalty[[i, bb]] = (num / den).powf(theta[bb]);
            }
        }

        cluster_sum.fill(0.0);
        for &j in cells {
            let mut sum = 0.0;
            for i in 0..k {
                let mut p = r_base[[i, j]];
                for ci in 0..n_cov {
                    p *= penalty[[i, design.global_level(ci, j)]];
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
                r[[i, j]] = rij;
                cluster_sum[i] += rij;
                for ci in 0..n_cov {
                    let bb = design.global_level(ci, j);
                    o[[i, bb]] = (o[[i, bb]] + rij).max(0.0);
                }
            }
        }
        for i in 0..k {
            let s = cluster_sum[i];
            for bb in 0..b {
                e[[i, bb]] = (e[[i, bb]] + s * design.pr_b[bb]).max(0.0);
            }
        }
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
    use rand::SeedableRng;
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
        rng: &mut impl Rng,
    ) {
        let n = r.ncols();
        let k = r.nrows();
        let b = design.n_levels;
        let n_cov = design.n_covariates;
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(rng);

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
                        diff < 1e-9,
                        "R mismatch at [{i},{j}] n_cov={n_cov}: {diff}"
                    );
                }
            }
            for i in 0..o.nrows() {
                for bb in 0..o.ncols() {
                    assert!((o_fast[[i, bb]] - o_slow[[i, bb]]).abs() < 1e-8);
                    assert!((e_fast[[i, bb]] - e_slow[[i, bb]]).abs() < 1e-8);
                }
            }
            for j in 0..r.ncols() {
                let col: f64 = (0..r.nrows()).map(|i| r_fast[[i, j]]).sum();
                assert!((col - 1.0).abs() < 1e-8, "column {j} summed to {col}");
            }
        }
    }

    /// Previous k-means++ that recomputed every cell-to-centroid distance for
    /// every new seed.
    fn reference_kmeans_cosine(
        z: ArrayView2<'_, f64>,
        k: usize,
        rng: &mut impl Rng,
        iters: usize,
    ) -> Array2<f64> {
        let d = z.nrows();
        let n = z.ncols();
        let k = k.min(n).max(1);
        let mut y = Array2::<f64>::zeros((d, k));
        let first = rng.gen_range(0..n);
        y.column_mut(0).assign(&z.column(first));
        let mut chosen = vec![first];

        for c in 1..k {
            let mut min_dist = vec![f64::INFINITY; n];
            for j in 0..n {
                for prev in 0..c {
                    let mut sim = 0.0;
                    for i in 0..d {
                        sim += y[[i, prev]] * z[[i, j]];
                    }
                    let dist = (2.0 * (1.0 - sim)).abs().max(1e-12);
                    if dist < min_dist[j] {
                        min_dist[j] = dist;
                    }
                }
            }
            let mut best_idx = 0usize;
            let mut best_score = f64::INFINITY;
            for j in 0..n {
                if chosen.contains(&j) {
                    continue;
                }
                let u: f64 = rng.gen::<f64>().clamp(1e-12, 1.0);
                let score = -u.ln() / min_dist[j];
                if score < best_score {
                    best_score = score;
                    best_idx = j;
                }
            }
            chosen.push(best_idx);
            y.column_mut(c).assign(&z.column(best_idx));
        }

        y = l2_normalize_columns(y.view());
        for _ in 0..iters {
            let dist = cosine_distance(y.view(), z);
            let mut counts = vec![0.0f64; k];
            let mut next = Array2::<f64>::zeros((d, k));
            for j in 0..n {
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..k {
                    let dval = dist[[c, j]];
                    if dval < best_d {
                        best_d = dval;
                        best = c;
                    }
                }
                counts[best] += 1.0;
                for i in 0..d {
                    next[[i, best]] += z[[i, j]];
                }
            }
            for c in 0..k {
                if counts[c] > 0.0 {
                    for i in 0..d {
                        next[[i, c]] /= counts[c];
                    }
                } else {
                    next.column_mut(c).assign(&y.column(c));
                }
            }
            y = l2_normalize_columns(next.view());
        }
        y
    }

    #[test]
    fn incremental_kmeans_pp_matches_nested_distance_scan() {
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

        let fast = kmeans_cosine(z.view(), k, &mut ChaCha8Rng::seed_from_u64(11), 6);
        let slow = reference_kmeans_cosine(z.view(), k, &mut ChaCha8Rng::seed_from_u64(11), 6);
        assert_eq!(fast.nrows(), slow.nrows());
        assert_eq!(fast.ncols(), slow.ncols());
        for i in 0..fast.nrows() {
            for j in 0..fast.ncols() {
                let diff = (fast[[i, j]] - slow[[i, j]]).abs();
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
