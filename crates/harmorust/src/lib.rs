//! Rust port of the Harmony single-cell integration algorithm.
//!
//! This is the same idea as [seurust](https://github.com/NebilI/seurust): keep
//! the published Harmony API (`RunHarmony` on a PCA embedding + batch labels)
//! and replace the C++ hot path (`src/harmony.cpp` in
//! [immunogenomics/harmony](https://github.com/immunogenomics/harmony)) with
//! native Rust.
//!
//! Harmony only needs the PCA cell embeddings that Seurat / seurust already
//! produce. A typical drop-in workflow is:
//!
//! ```text
//! seurust  →  Normalize / scale / SNN kernels
//! Seurat   →  RunPCA
//! harmorust →  RunHarmony on those PCs
//! seurust  →  neighbors / clustering on the harmony reduction
//! ```

pub mod binio;
pub mod cluster;
pub mod compare;
pub mod config;
pub mod correct;
pub mod design;
pub mod error;
pub mod harmony;
pub mod linalg;
pub mod metrics;
mod vectormath;
pub mod parity;
pub mod r_rng;
mod reference;
pub mod simulate;
pub mod timers;

pub use config::{r_round_half_even, HarmonyConfig};
pub use error::{HarmonyError, Result};
pub use harmony::{embeddings_from_rows, run_harmony, run_harmony_multi, HarmonyResult, HarmonyState};
pub use metrics::{evaluate, IntegrationMetrics};
pub use parity::{assert_parity_within, parity_max_abs, PARITY_MAX_ABS};
pub use simulate::{simulate_atlas, SimulateSpec, SimulatedAtlas};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::rows_to_array;

    fn toy_atlas() -> (SimulatedAtlas, HarmonyConfig) {
        let spec = SimulateSpec {
            n_cells: 240,
            n_pcs: 6,
            n_batches: 2,
            n_types: 3,
            batch_strength: 2.8,
            seed: 11,
        };
        let mut cfg = HarmonyConfig::default();
        cfg.nclust = Some(3);
        cfg.max_iter = 10;
        cfg.max_iter_cluster = 8;
        cfg.seed = 11;
        cfg.lambda = Some(1.0);
        cfg.sigma = 0.2;
        cfg.early_stop = false;
        (simulate_atlas(&spec), cfg)
    }

    #[test]
    fn one_cluster_is_linear_batch_regression() {
        let (data, mut cfg) = toy_atlas();
        cfg.nclust = Some(1);
        cfg.theta = 0.0;
        let z = embeddings_from_rows(&data.embeddings).unwrap();
        let before = evaluate(z.view(), &data.batch, &data.cell_type);
        let out = run_harmony(z.view(), &data.batch, &cfg).unwrap();
        let after = evaluate(rows_to_array(&out.embeddings).view(), &data.batch, &data.cell_type);
        eprintln!(
            "nclust=1  batch {} → {}  type {} → {}",
            before.batch_separation, after.batch_separation, before.type_separation, after.type_separation
        );
        assert!(
            after.batch_separation < before.batch_separation * 0.25,
            "linear ridge should almost remove a global batch shift: {} → {}",
            before.batch_separation,
            after.batch_separation
        );
    }

    #[test]
    fn batch_shift_shrinks_and_types_remain() {
        let (data, cfg) = toy_atlas();
        let z = embeddings_from_rows(&data.embeddings).unwrap();
        let before = evaluate(z.view(), &data.batch, &data.cell_type);

        let out = run_harmony(z.view(), &data.batch, &cfg).unwrap();
        let after_z = rows_to_array(&out.embeddings);
        let after = evaluate(after_z.view(), &data.batch, &data.cell_type);
        eprintln!(
            "iters={} clusters={} batch {} → {} (lisi {} → {}) type {} → {} (lisi {} → {})",
            out.n_iter,
            out.n_clusters,
            before.batch_separation,
            after.batch_separation,
            before.batch_lisi,
            after.batch_lisi,
            before.type_separation,
            after.type_separation,
            before.type_lisi,
            after.type_lisi
        );

        assert!(
            after.batch_separation < before.batch_separation * 0.15,
            "batch separation did not drop enough: {} → {}",
            before.batch_separation,
            after.batch_separation
        );
        assert!(
            after.batch_lisi > 1.6,
            "batch LISI did not rise: {} → {}",
            before.batch_lisi,
            after.batch_lisi
        );
        assert!(
            (after.type_separation - before.type_separation).abs()
                < before.type_separation * 0.15,
            "cell-type geometry changed too much: {} → {}",
            before.type_separation,
            after.type_separation
        );
        assert!(out.n_iter >= 1);
        assert_eq!(out.embeddings.len(), data.n_cells);
        assert_eq!(out.embeddings[0].len(), data.n_pcs);
    }

    #[test]
    fn two_covariates_run() {
        let spec = SimulateSpec {
            n_cells: 120,
            n_pcs: 4,
            n_batches: 2,
            n_types: 3,
            batch_strength: 1.6,
            seed: 3,
        };
        let data = simulate_atlas(&spec);
        let z = embeddings_from_rows(&data.embeddings).unwrap();
        let donor = data.batch.clone();
        let protocol: Vec<String> = (0..data.n_cells)
            .map(|i| if i % 2 == 0 { "v2".into() } else { "v3".into() })
            .collect();
        let mut cfg = HarmonyConfig::default();
        cfg.nclust = Some(4);
        cfg.max_iter = 4;
        let out = run_harmony_multi(
            z.view(),
            &[donor, protocol],
            &[2.0, 1.0],
            Some(&[1.0, 1.0]),
            &cfg,
        )
        .unwrap();
        assert_eq!(out.embeddings.len(), data.n_cells);
    }

    #[test]
    fn rejects_tiny_matrices() {
        let z = embeddings_from_rows(&vec![vec![0.1, 0.2]; 4]).unwrap();
        let labels = vec!["a", "b", "a", "b"];
        let err = run_harmony(z.view(), &labels, &HarmonyConfig::default()).unwrap_err();
        assert!(err.to_string().contains("fewer than 6"));
    }

    #[test]
    fn thread_count_does_not_change_bits() {
        let spec = SimulateSpec {
            n_cells: 90,
            n_pcs: 4,
            n_batches: 2,
            n_types: 3,
            batch_strength: 2.0,
            seed: 5,
        };
        let data = simulate_atlas(&spec);
        let z = embeddings_from_rows(&data.embeddings).unwrap();
        let mut cfg = HarmonyConfig::default();
        cfg.nclust = Some(4);
        cfg.max_iter = 3;
        cfg.seed = 99;
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| run_harmony(z.view(), &data.batch, &cfg).unwrap())
        };
        let a = run(1);
        let b = run(4);
        assert_eq!(a.embeddings, b.embeddings);
        assert_eq!(a.r, b.r);
        assert_eq!(a.n_iter, b.n_iter);
    }

    #[test]
    fn seeded_runs_are_deterministic() {
        let spec = SimulateSpec {
            n_cells: 90,
            n_pcs: 4,
            n_batches: 2,
            n_types: 3,
            batch_strength: 2.0,
            seed: 5,
        };
        let data = simulate_atlas(&spec);
        let z = embeddings_from_rows(&data.embeddings).unwrap();
        let mut cfg = HarmonyConfig::default();
        cfg.nclust = Some(4);
        cfg.max_iter = 3;
        cfg.seed = 99;
        let a = run_harmony(z.view(), &data.batch, &cfg).unwrap();
        let b = run_harmony(z.view(), &data.batch, &cfg).unwrap();
        assert_eq!(a.embeddings, b.embeddings);
        assert_eq!(a.n_iter, b.n_iter);
    }

    #[test]
    fn assignments_are_a_simplex_and_auto_lambda_runs() {
        let (data, mut cfg) = toy_atlas();
        cfg.lambda = None;
        cfg.max_iter = 4;
        let z = embeddings_from_rows(&data.embeddings).unwrap();
        let out = run_harmony(z.view(), &data.batch, &cfg).unwrap();
        assert_eq!(out.r.len(), out.n_clusters);
        for j in 0..data.n_cells {
            let col: f64 = out.r.iter().map(|row| row[j]).sum();
            assert!((col - 1.0).abs() < 1e-6, "R column {j} summed to {col}");
        }
    }
}
