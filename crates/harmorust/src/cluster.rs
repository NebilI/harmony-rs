use crate::design::Design;
use crate::linalg::{column_l1_normalize, cosine_distance, l2_normalize_columns, softmax_neg_dist};
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
        // Harmony's weighted sample: pick argmin of -log(U) / dist
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
pub fn update_r(
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

pub fn objective(
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
