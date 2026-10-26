//! Golden-fixture regression tests that do not need CRAN Harmony.
//!
//! `scripts/regen-fixtures.sh` (or `HARMORUST_REGEN_FIXTURES=1 cargo test`)
//! rewrites the JSON when the faithful kernel changes on purpose.

use harmorust::{
    embeddings_from_rows, run_harmony, run_harmony_multi, simulate_atlas, HarmonyConfig,
    HarmonyResult, SimulateSpec,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
struct Fixture {
    spec: SimulateSpec,
    nclust: usize,
    max_iter: usize,
    seed: u64,
    result: HarmonyResult,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn run_case(spec: &SimulateSpec, cfg: &HarmonyConfig) -> HarmonyResult {
    let atlas = simulate_atlas(spec);
    let z = embeddings_from_rows(&atlas.embeddings).unwrap();
    run_harmony(z.view(), &atlas.batch, cfg).unwrap()
}

fn load_or_write(name: &str, spec: SimulateSpec, nclust: usize, max_iter: usize) -> Fixture {
    let path = fixture_dir().join(name);
    let cfg = HarmonyConfig {
        nclust: Some(nclust),
        max_iter,
        max_iter_cluster: 4,
        early_stop: false,
        seed: spec.seed,
        lambda: Some(1.0),
        ..HarmonyConfig::default()
    };
    let got = run_case(&spec, &cfg);
    let regen = std::env::var("HARMORUST_REGEN_FIXTURES").is_ok();
    if regen || !path.exists() {
        let fx = Fixture {
            spec: spec.clone(),
            nclust,
            max_iter,
            seed: spec.seed,
            result: got.clone(),
        };
        if regen {
            std::fs::create_dir_all(fixture_dir()).unwrap();
            std::fs::write(&path, serde_json::to_string_pretty(&fx).unwrap()).unwrap();
        }
        return fx;
    }
    let raw = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn max_abs(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

fn assert_same(got: &HarmonyResult, want: &HarmonyResult) {
    assert_eq!(got.n_clusters, want.n_clusters);
    assert_eq!(got.n_iter, want.n_iter);
    // JSON f64 round-trip is not bitwise; keep a few ulp.
    for (name, g, w) in [
        ("embeddings", &got.embeddings, &want.embeddings),
        ("r", &got.r, &want.r),
        ("y", &got.y, &want.y),
        ("o", &got.o, &want.o),
        ("e", &got.e, &want.e),
    ] {
        let d = max_abs(g, w);
        assert!(d < 1e-12, "{name} max |Δ|={d}");
    }
}

#[test]
fn fixture_atlas_80_k3_is_stable() {
    let spec = SimulateSpec {
        n_cells: 80,
        n_pcs: 6,
        n_batches: 2,
        n_types: 3,
        batch_strength: 1.6,
        seed: 3,
    };
    let fx = load_or_write("atlas-80-k3.json", spec.clone(), 3, 3);
    let cfg = HarmonyConfig {
        nclust: Some(3),
        max_iter: 3,
        max_iter_cluster: 4,
        early_stop: false,
        seed: 3,
        lambda: Some(1.0),
        ..HarmonyConfig::default()
    };
    let got = run_case(&spec, &cfg);
    if fixture_dir().join("atlas-80-k3.json").exists()
        && std::env::var("HARMORUST_REGEN_FIXTURES").is_err()
    {
        assert_same(&got, &fx.result);
    } else {
        assert_eq!(got.n_clusters, 3);
        assert_eq!(got.embeddings.len(), 80);
        assert_eq!(got.e.len(), got.n_clusters);
    }
}

#[test]
fn fixture_tiny_n_below_40() {
    let spec = SimulateSpec {
        n_cells: 36,
        n_pcs: 4,
        n_batches: 2,
        n_types: 3,
        batch_strength: 1.2,
        seed: 9,
    };
    let cfg = HarmonyConfig {
        nclust: Some(3),
        max_iter: 2,
        early_stop: false,
        seed: 9,
        ..HarmonyConfig::default()
    };
    let out = run_case(&spec, &cfg);
    assert_eq!(out.embeddings.len(), 36);
    assert_eq!(out.e.len(), out.n_clusters);
}

#[test]
fn fixture_auto_lambda_and_tau() {
    let spec = SimulateSpec {
        n_cells: 90,
        n_pcs: 8,
        n_batches: 3,
        n_types: 3,
        batch_strength: 1.4,
        seed: 4,
    };
    let cfg = HarmonyConfig {
        nclust: Some(5),
        max_iter: 2,
        early_stop: false,
        seed: 4,
        lambda: None,
        tau: 2.0,
        ..HarmonyConfig::default()
    };
    let out = run_case(&spec, &cfg);
    assert_eq!(out.embeddings.len(), 90);
}

#[test]
fn fixture_two_covariates_and_dropped_batch() {
    let spec = SimulateSpec {
        n_cells: 60,
        n_pcs: 5,
        n_batches: 4,
        n_types: 2,
        batch_strength: 2.2,
        seed: 8,
    };
    let atlas = simulate_atlas(&spec);
    let z = embeddings_from_rows(&atlas.embeddings).unwrap();
    let proto: Vec<String> = (0..atlas.n_cells)
        .map(|i| if i % 2 == 0 { "v2".into() } else { "v3".into() })
        .collect();
    let cfg = HarmonyConfig {
        nclust: Some(4),
        max_iter: 2,
        early_stop: false,
        seed: 8,
        batch_prop_cutoff: 0.4,
        ..HarmonyConfig::default()
    };
    let out = run_harmony_multi(
        z.view(),
        &[atlas.batch.clone(), proto],
        &[2.0, 1.0],
        Some(&[1.0, 1.0]),
        &cfg,
    )
    .unwrap();
    assert_eq!(out.embeddings.len(), 60);
}

#[test]
fn fixture_50_pcs() {
    let spec = SimulateSpec {
        n_cells: 48,
        n_pcs: 50,
        n_batches: 2,
        n_types: 3,
        batch_strength: 1.1,
        seed: 2,
    };
    let cfg = HarmonyConfig {
        nclust: Some(3),
        max_iter: 2,
        early_stop: false,
        seed: 2,
        lambda: Some(1.0),
        ..HarmonyConfig::default()
    };
    let out = run_case(&spec, &cfg);
    assert_eq!(out.embeddings[0].len(), 50);
}

#[test]
fn fixture_unbalanced_early_stop() {
    let spec = SimulateSpec {
        n_cells: 70,
        n_pcs: 6,
        n_batches: 3,
        n_types: 2,
        batch_strength: 3.0,
        seed: 6,
    };
    let cfg = HarmonyConfig {
        nclust: Some(4),
        max_iter: 6,
        early_stop: true,
        seed: 6,
        ..HarmonyConfig::default()
    };
    let out = run_case(&spec, &cfg);
    assert!(out.n_iter >= 1);
    assert_eq!(out.embeddings.len(), 70);
}
