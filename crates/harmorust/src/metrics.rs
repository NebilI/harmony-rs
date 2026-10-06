use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    pub batch_separation: f64,
    pub type_separation: f64,
    pub batch_lisi: f64,
    pub type_lisi: f64,
}

/// Cheap proxies for "did Harmony work?":
///
/// * `*_separation` — mean distance of each group centroid from the global
///   centroid. Batch separation should drop; type separation should stay.
/// * `*_lisi` — a local inverse Simpson index on 30 nearest neighbours in the
///   first two PCs. Batch LISI should rise toward the number of batches.
pub fn evaluate(
    embeddings: ArrayView2<'_, f64>,
    batch: &[String],
    cell_type: &[String],
) -> IntegrationMetrics {
    IntegrationMetrics {
        batch_separation: group_separation(embeddings, batch),
        type_separation: group_separation(embeddings, cell_type),
        batch_lisi: mean_lisi(embeddings, batch, 30),
        type_lisi: mean_lisi(embeddings, cell_type, 30),
    }
}

fn group_separation(z: ArrayView2<'_, f64>, labels: &[String]) -> f64 {
    let n = z.nrows();
    if n == 0 {
        return 0.0;
    }
    let d = z.ncols().min(4);
    let mut global = vec![0.0; d];
    for i in 0..n {
        for j in 0..d {
            global[j] += z[[i, j]];
        }
    }
    for g in global.iter_mut() {
        *g /= n as f64;
    }

    let mut sums: HashMap<&str, (Vec<f64>, usize)> = HashMap::new();
    for i in 0..n {
        let entry = sums
            .entry(labels[i].as_str())
            .or_insert_with(|| (vec![0.0; d], 0));
        for j in 0..d {
            entry.0[j] += z[[i, j]];
        }
        entry.1 += 1;
    }
    let mut sep = 0.0;
    for (mean, count) in sums.values() {
        let mut dist = 0.0;
        for j in 0..d {
            let m = mean[j] / *count as f64;
            dist += (m - global[j]).powi(2);
        }
        sep += dist.sqrt();
    }
    sep / sums.len() as f64
}

fn mean_lisi(z: ArrayView2<'_, f64>, labels: &[String], k: usize) -> f64 {
    let n = z.nrows();
    if n < 3 {
        return 1.0;
    }
    let d = z.ncols().min(2);
    let k = k.min(n - 1).max(1);
    let mut acc = 0.0;
    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut dist = 0.0;
            for p in 0..d {
                let diff = z[[i, p]] - z[[j, p]];
                dist += diff * diff;
            }
            dists.push((dist, j));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut counts: HashMap<&str, f64> = HashMap::new();
        for &(_, j) in dists.iter().take(k) {
            *counts.entry(labels[j].as_str()).or_insert(0.0) += 1.0;
        }
        let mut simpson = 0.0;
        for c in counts.values() {
            let p = *c / k as f64;
            simpson += p * p;
        }
        acc += if simpson > 0.0 { 1.0 / simpson } else { 1.0 };
    }
    acc / n as f64
}

pub fn rows_to_array(rows: &[Vec<f64>]) -> Array2<f64> {
    let n = rows.len();
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut z = Array2::<f64>::zeros((n, d));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            z[[i, j]] = v;
        }
    }
    z
}
