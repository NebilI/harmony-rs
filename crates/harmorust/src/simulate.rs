use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedAtlas {
    pub embeddings: Vec<Vec<f64>>,
    pub batch: Vec<String>,
    pub cell_type: Vec<String>,
    pub n_cells: usize,
    pub n_pcs: usize,
    pub n_batches: usize,
    pub n_types: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulateSpec {
    pub n_cells: usize,
    pub n_pcs: usize,
    pub n_batches: usize,
    pub n_types: usize,
    /// How far batches are shifted along PC1/PC2, relative to cell-type spread.
    pub batch_strength: f64,
    pub seed: u64,
}

impl Default for SimulateSpec {
    fn default() -> Self {
        Self {
            n_cells: 600,
            n_pcs: 8,
            n_batches: 2,
            n_types: 3,
            batch_strength: 2.4,
            seed: 7,
        }
    }
}

/// Build a small PBMC-style atlas: shared cell-type geometry plus a batch shift.
///
/// This is the same kind of synthetic test Harmony (and seurust integration
/// demos) use — cell types stay separable, batches do not overlap until
/// correction.
pub fn simulate_atlas(spec: &SimulateSpec) -> SimulatedAtlas {
    let mut rng = ChaCha8Rng::seed_from_u64(spec.seed);
    let n = spec.n_cells.max(12);
    let d = spec.n_pcs.max(2);
    let n_types = spec.n_types.max(2);
    let n_batches = spec.n_batches.max(2);

    let type_names = ["T cell", "B cell", "Monocyte", "NK cell", "DC"];
    let batch_names = ["donor_A", "donor_B", "donor_C", "donor_D"];

    let mut centers = Array2::<f64>::zeros((n_types, d));
    for t in 0..n_types {
        let angle = t as f64 * std::f64::consts::TAU / n_types as f64;
        centers[[t, 0]] = 3.2 * angle.cos();
        centers[[t, 1]] = 3.2 * angle.sin();
        for j in 2..d {
            centers[[t, j]] = rng.gen_range(-0.4..0.4);
        }
    }

    let mut z = Array2::<f64>::zeros((n, d));
    let mut batch = Vec::with_capacity(n);
    let mut cell_type = Vec::with_capacity(n);

    for i in 0..n {
        let t = i % n_types;
        let b = (i / n_types) % n_batches;
        let batch_angle = b as f64 * std::f64::consts::TAU / n_batches as f64;
        let shift0 = spec.batch_strength * batch_angle.cos();
        let shift1 = spec.batch_strength * batch_angle.sin();

        z[[i, 0]] = centers[[t, 0]] + shift0 + rng.gen_range(-0.35..0.35);
        z[[i, 1]] = centers[[t, 1]] + shift1 + rng.gen_range(-0.35..0.35);
        for j in 2..d {
            z[[i, j]] = centers[[t, j]] + rng.gen_range(-0.25..0.25);
        }

        batch.push(
            batch_names
                .get(b)
                .copied()
                .unwrap_or("donor")
                .to_string(),
        );
        cell_type.push(
            type_names
                .get(t)
                .copied()
                .unwrap_or("other")
                .to_string(),
        );
    }

    let embeddings = (0..n).map(|i| (0..d).map(|j| z[[i, j]]).collect()).collect();
    SimulatedAtlas {
        embeddings,
        batch,
        cell_type,
        n_cells: n,
        n_pcs: d,
        n_batches,
        n_types,
    }
}
