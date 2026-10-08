//! Repeatable wall-time comparison of harmorust vs immunogenomics/harmony.
//!
//! Both implementations see the same synthetic atlas and the same Harmony
//! hyperparameters. Early stopping is off so each side does a fixed amount of
//! work (`max_iter` outer rounds). Reported times are medians over repeated
//! runs after a warmup.

use crate::config::HarmonyConfig;
use crate::error::{HarmonyError, Result};
use crate::harmony::{embeddings_from_rows, run_harmony};
use crate::simulate::{simulate_atlas, SimulateSpec, SimulatedAtlas};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const ORIGINAL_R: &str = include_str!("../scripts/original_harmony.R");

#[derive(Debug, Clone)]
pub struct CompareProtocol {
    pub cell_counts: Vec<usize>,
    pub n_pcs: usize,
    pub n_batches: usize,
    pub n_types: usize,
    pub batch_strength: f64,
    pub seed: u64,
    pub warmup: usize,
    pub repeats: usize,
    pub max_iter: usize,
    pub max_iter_cluster: usize,
    pub ncores: usize,
    pub early_stop: bool,
    pub theta: f64,
    pub sigma: f64,
    pub lambda: f64,
}

impl Default for CompareProtocol {
    fn default() -> Self {
        Self {
            cell_counts: vec![500, 1000, 2000, 4000, 8000],
            n_pcs: 20,
            n_batches: 4,
            n_types: 5,
            batch_strength: 1.8,
            seed: 1,
            warmup: 2,
            repeats: 5,
            max_iter: 8,
            max_iter_cluster: 4,
            ncores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            early_stop: false,
            theta: 2.0,
            sigma: 0.1,
            lambda: 1.0,
        }
    }
}

impl CompareProtocol {
    pub fn cases(&self) -> Vec<BenchCase> {
        self.cell_counts
            .iter()
            .copied()
            .map(|n_cells| BenchCase {
                name: format!("atlas-{n_cells}"),
                spec: SimulateSpec {
                    n_cells,
                    n_pcs: self.n_pcs,
                    n_batches: self.n_batches,
                    n_types: self.n_types,
                    batch_strength: self.batch_strength,
                    seed: self.seed,
                },
                nclust: self.n_types,
            })
            .collect()
    }

    pub fn config(&self, nclust: usize) -> HarmonyConfig {
        HarmonyConfig {
            theta: self.theta,
            sigma: self.sigma,
            lambda: Some(self.lambda),
            nclust: Some(nclust),
            max_iter: self.max_iter,
            max_iter_cluster: self.max_iter_cluster,
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
            block_size: 0.05,
            alpha: 0.2,
            tau: 0.0,
            batch_prop_cutoff: 1e-5,
            seed: self.seed,
            early_stop: self.early_stop,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchCase {
    pub name: String,
    pub spec: SimulateSpec,
    pub nclust: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimingStats {
    pub times_ms: Vec<f64>,
    pub median_ms: f64,
    pub mad_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub n_iter: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareRow {
    pub name: String,
    pub n_cells: usize,
    pub n_pcs: usize,
    pub n_batches: usize,
    pub n_clusters: usize,
    pub original: Option<TimingStats>,
    pub rust: TimingStats,
    pub original_version: Option<String>,
}

impl CompareRow {
    /// Original median / harmorust median. Values > 1 mean harmorust is faster.
    pub fn speedup(&self) -> Option<f64> {
        let original = self.original.as_ref()?;
        if self.rust.median_ms <= 0.0 {
            return None;
        }
        Some(original.median_ms / self.rust.median_ms)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareReport {
    pub protocol: ProtocolSummary,
    pub original_version: Option<String>,
    pub rows: Vec<CompareRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolSummary {
    pub n_pcs: usize,
    pub n_batches: usize,
    pub n_types: usize,
    pub warmup: usize,
    pub repeats: usize,
    pub max_iter: usize,
    pub max_iter_cluster: usize,
    pub ncores: usize,
    pub early_stop: bool,
    pub theta: f64,
    pub sigma: f64,
    pub lambda: f64,
    pub seed: u64,
}

impl From<&CompareProtocol> for ProtocolSummary {
    fn from(p: &CompareProtocol) -> Self {
        Self {
            n_pcs: p.n_pcs,
            n_batches: p.n_batches,
            n_types: p.n_types,
            warmup: p.warmup,
            repeats: p.repeats,
            max_iter: p.max_iter,
            max_iter_cluster: p.max_iter_cluster,
            ncores: p.ncores,
            early_stop: p.early_stop,
            theta: p.theta,
            sigma: p.sigma,
            lambda: p.lambda,
            seed: p.seed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompareOptions {
    pub rust_only: bool,
    pub rscript: PathBuf,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            rust_only: false,
            rscript: PathBuf::from("Rscript"),
        }
    }
}

pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

/// Median absolute deviation, a robust spread estimate.
pub fn mad(xs: &[f64], med: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let devs: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
    median(&devs)
}

pub fn stats_from_times(times_ms: Vec<f64>, n_iter: usize) -> TimingStats {
    let median_ms = median(&times_ms);
    TimingStats {
        mad_ms: mad(&times_ms, median_ms),
        min_ms: times_ms.iter().copied().fold(f64::INFINITY, f64::min),
        max_ms: times_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        median_ms,
        times_ms,
        n_iter,
    }
}

pub fn fmt_ms(ms: f64) -> String {
    if !ms.is_finite() {
        return "n/a".into();
    }
    if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else if ms >= 10.0 {
        format!("{:.1} ms", ms)
    } else {
        format!("{:.2} ms", ms)
    }
}

pub fn fmt_speedup(speedup: Option<f64>) -> String {
    match speedup {
        Some(x) if x.is_finite() => format!("{x:.2}×"),
        _ => "n/a".into(),
    }
}

pub fn format_markdown_table(rows: &[CompareRow]) -> String {
    let mut out = String::new();
    out.push_str(
        "| Case | Cells | PCs | Batches | Clusters | Original Harmony | harmorust | Speedup |\n",
    );
    out.push_str(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for row in rows {
        let original = row
            .original
            .as_ref()
            .map(|s| fmt_ms(s.median_ms))
            .unwrap_or_else(|| "n/a".into());
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.name,
            row.n_cells,
            row.n_pcs,
            row.n_batches,
            row.n_clusters,
            original,
            fmt_ms(row.rust.median_ms),
            fmt_speedup(row.speedup()),
        ));
    }
    out
}

pub fn geometric_mean_speedup(rows: &[CompareRow]) -> Option<f64> {
    let xs: Vec<f64> = rows.iter().filter_map(CompareRow::speedup).collect();
    if xs.is_empty() {
        return None;
    }
    let log_sum: f64 = xs.iter().map(|x| x.ln()).sum();
    Some((log_sum / xs.len() as f64).exp())
}

pub fn time_harmorust(
    atlas: &SimulatedAtlas,
    cfg: &HarmonyConfig,
    warmup: usize,
    repeats: usize,
) -> Result<TimingStats> {
    if repeats == 0 {
        return Err(HarmonyError::msg("repeats must be >= 1"));
    }
    let z = embeddings_from_rows(&atlas.embeddings)?;
    for _ in 0..warmup {
        let _ = run_harmony(z.view(), &atlas.batch, cfg)?;
    }
    let mut times_ms = Vec::with_capacity(repeats);
    let mut n_iter = 0usize;
    for _ in 0..repeats {
        let start = Instant::now();
        let out = run_harmony(z.view(), &atlas.batch, cfg)?;
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        n_iter = out.n_iter;
    }
    Ok(stats_from_times(times_ms, n_iter))
}

#[derive(Debug, Deserialize)]
struct OriginalJson {
    version: String,
    n_iter: usize,
    times_ms: Vec<f64>,
}

pub fn time_original_harmony(
    atlas: &SimulatedAtlas,
    protocol: &CompareProtocol,
    nclust: usize,
    rscript: &Path,
    workdir: &Path,
) -> Result<(TimingStats, String)> {
    fs::create_dir_all(workdir).map_err(|e| HarmonyError::msg(e.to_string()))?;
    let script = workdir.join("original_harmony.R");
    fs::write(&script, ORIGINAL_R).map_err(|e| HarmonyError::msg(e.to_string()))?;
    let emb_path = workdir.join("embeddings.csv");
    let batch_path = workdir.join("batch.csv");
    let out_path = workdir.join("original.json");
    write_embeddings_csv(&emb_path, &atlas.embeddings)?;
    write_batch_csv(&batch_path, &atlas.batch)?;

    let mut cmd = Command::new(rscript);
    cmd.arg(&script)
        .arg("--embeddings")
        .arg(&emb_path)
        .arg("--batch")
        .arg(&batch_path)
        .arg("--nclust")
        .arg(nclust.to_string())
        .arg("--max-iter")
        .arg(protocol.max_iter.to_string())
        .arg("--warmup")
        .arg(protocol.warmup.to_string())
        .arg("--repeats")
        .arg(protocol.repeats.to_string())
        .arg("--ncores")
        .arg(protocol.ncores.to_string())
        .arg("--early-stop")
        .arg(if protocol.early_stop { "TRUE" } else { "FALSE" })
        .arg("--theta")
        .arg(protocol.theta.to_string())
        .arg("--sigma")
        .arg(protocol.sigma.to_string())
        .arg("--lambda")
        .arg(protocol.lambda.to_string())
        .arg("--max-iter-cluster")
        .arg(protocol.max_iter_cluster.to_string())
        .arg("--seed")
        .arg(protocol.seed.to_string())
        .arg("--out")
        .arg(&out_path);
    if let Some(lib) = r_user_lib() {
        cmd.env("R_LIBS_USER", lib);
    }

    let output = cmd
        .output()
        .map_err(|e| HarmonyError::msg(format!("failed to spawn Rscript: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(HarmonyError::msg(format!(
            "original Harmony (R) failed ({}):\n{stderr}{stdout}",
            output.status
        )));
    }
    let raw = fs::read_to_string(&out_path).map_err(|e| {
        HarmonyError::msg(format!(
            "original Harmony wrote no timings ({e}). R stdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        ))
    })?;
    let parsed: OriginalJson = serde_json::from_str(raw.trim()).map_err(|e| {
        HarmonyError::msg(format!("could not parse original Harmony JSON: {e}\n{raw}"))
    })?;
    if parsed.times_ms.len() != protocol.repeats {
        return Err(HarmonyError::msg(format!(
            "original Harmony returned {} timings, expected {}",
            parsed.times_ms.len(),
            protocol.repeats
        )));
    }
    Ok((
        stats_from_times(parsed.times_ms, parsed.n_iter),
        parsed.version,
    ))
}

pub fn run_compare(
    protocol: &CompareProtocol,
    opts: &CompareOptions,
) -> Result<CompareReport> {
    if protocol.repeats == 0 {
        return Err(HarmonyError::msg("repeats must be >= 1"));
    }

    let pid = std::process::id();
    let mut rows = Vec::new();
    let mut original_version = None;
    for (i, case) in protocol.cases().into_iter().enumerate() {
        let atlas = simulate_atlas(&case.spec);
        let cfg = protocol.config(case.nclust);
        let rust = time_harmorust(&atlas, &cfg, protocol.warmup, protocol.repeats)?;
        let original = if opts.rust_only {
            None
        } else {
            let workdir = std::env::temp_dir().join(format!("harmorust-compare-{pid}-{i}"));
            let (stats, version) = time_original_harmony(
                &atlas,
                protocol,
                case.nclust,
                &opts.rscript,
                &workdir,
            )?;
            original_version = Some(version.clone());
            let _ = fs::remove_dir_all(&workdir);
            Some(stats)
        };
        rows.push(CompareRow {
            name: case.name,
            n_cells: atlas.n_cells,
            n_pcs: atlas.n_pcs,
            n_batches: atlas.n_batches,
            n_clusters: case.nclust,
            original,
            rust,
            original_version: original_version.clone(),
        });
    }
    Ok(CompareReport {
        protocol: ProtocolSummary::from(protocol),
        original_version,
        rows,
    })
}

pub fn print_report(report: &CompareReport, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(report).unwrap());
        return;
    }
    let p = &report.protocol;
    println!("Harmony kernel comparison");
    println!(
        "protocol: {} PCs, {} batches, {} cell types, max_iter={}, max_iter_cluster={}, early_stop={}, lambda={}, ncores={}",
        p.n_pcs,
        p.n_batches,
        p.n_types,
        p.max_iter,
        p.max_iter_cluster,
        p.early_stop,
        p.lambda,
        p.ncores
    );
    println!(
        "timing: {} warmup + {} measured runs, median wall time (MAD in parentheses)",
        p.warmup, p.repeats
    );
    if let Some(v) = &report.original_version {
        println!("original: immunogenomics/harmony {v} (C++ via R)");
    }
    println!("new: harmorust {}", env!("CARGO_PKG_VERSION"));
    println!();
    print!("{}", format_markdown_table(&report.rows));
    if let Some(geo) = geometric_mean_speedup(&report.rows) {
        println!("\ngeometric-mean speedup (original / harmorust): {:.2}×", geo);
    }
    println!();
    for row in &report.rows {
        let orig = row
            .original
            .as_ref()
            .map(|s| {
                format!(
                    "original median {} (MAD {}, n_iter {})",
                    fmt_ms(s.median_ms),
                    fmt_ms(s.mad_ms),
                    s.n_iter
                )
            })
            .unwrap_or_else(|| "original skipped".into());
        println!(
            "{}: {orig}; harmorust median {} (MAD {}, n_iter {})",
            row.name,
            fmt_ms(row.rust.median_ms),
            fmt_ms(row.rust.mad_ms),
            row.rust.n_iter
        );
    }
}

fn write_embeddings_csv(path: &Path, embeddings: &[Vec<f64>]) -> Result<()> {
    let mut f = fs::File::create(path).map_err(|e| HarmonyError::msg(e.to_string()))?;
    if embeddings.is_empty() {
        return Err(HarmonyError::msg("empty embedding"));
    }
    let d = embeddings[0].len();
    let header: Vec<String> = (0..d).map(|j| format!("pc{j}")).collect();
    writeln!(f, "{}", header.join(",")).map_err(|e| HarmonyError::msg(e.to_string()))?;
    for row in embeddings {
        let line = row
            .iter()
            .map(|v| format!("{v:.12}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(f, "{line}").map_err(|e| HarmonyError::msg(e.to_string()))?;
    }
    Ok(())
}

fn write_batch_csv(path: &Path, batch: &[String]) -> Result<()> {
    let mut f = fs::File::create(path).map_err(|e| HarmonyError::msg(e.to_string()))?;
    writeln!(f, "batch").map_err(|e| HarmonyError::msg(e.to_string()))?;
    for label in batch {
        writeln!(f, "{label}").map_err(|e| HarmonyError::msg(e.to_string()))?;
    }
    Ok(())
}

fn r_user_lib() -> Option<PathBuf> {
    if let Ok(lib) = std::env::var("R_LIBS_USER") {
        let p = PathBuf::from(lib);
        if p.join("harmony").exists() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join("R/library");
        if p.join("harmony").exists() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_even_and_empty() {
        assert!(median(&[]).is_nan());
        assert_eq!(median(&[3.0]), 3.0);
        assert_eq!(median(&[1.0, 3.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn mad_is_zero_when_all_equal() {
        assert_eq!(mad(&[5.0, 5.0, 5.0], 5.0), 0.0);
        assert_eq!(mad(&[1.0, 2.0, 3.0], 2.0), 1.0);
    }

    #[test]
    fn speedup_is_original_over_rust() {
        let row = CompareRow {
            name: "t".into(),
            n_cells: 10,
            n_pcs: 2,
            n_batches: 2,
            n_clusters: 2,
            original: Some(stats_from_times(vec![20.0, 22.0, 21.0], 8)),
            rust: stats_from_times(vec![10.0, 10.0, 10.0], 8),
            original_version: Some("2.0.5".into()),
        };
        assert!((row.speedup().unwrap() - 2.1).abs() < 1e-9);
        assert_eq!(fmt_speedup(row.speedup()), "2.10×");
    }

    #[test]
    fn markdown_table_lists_both_impls_and_speedup() {
        let row = CompareRow {
            name: "atlas-500".into(),
            n_cells: 500,
            n_pcs: 20,
            n_batches: 4,
            n_clusters: 5,
            original: Some(stats_from_times(vec![100.0], 8)),
            rust: stats_from_times(vec![25.0], 8),
            original_version: Some("2.0.5".into()),
        };
        let table = format_markdown_table(&[row]);
        assert!(table.contains("Original Harmony"));
        assert!(table.contains("harmorust"));
        assert!(table.contains("Speedup"));
        assert!(table.contains("4.00×"));
        assert!(table.contains("atlas-500"));
    }

    #[test]
    fn geometric_mean_of_speedups() {
        let a = CompareRow {
            name: "a".into(),
            n_cells: 1,
            n_pcs: 1,
            n_batches: 2,
            n_clusters: 2,
            original: Some(stats_from_times(vec![4.0], 1)),
            rust: stats_from_times(vec![1.0], 1),
            original_version: None,
        };
        let b = CompareRow {
            name: "b".into(),
            n_cells: 1,
            n_pcs: 1,
            n_batches: 2,
            n_clusters: 2,
            original: Some(stats_from_times(vec![9.0], 1)),
            rust: stats_from_times(vec![1.0], 1),
            original_version: None,
        };
        let geo = geometric_mean_speedup(&[a, b]).unwrap();
        assert!((geo - 6.0).abs() < 1e-9);
    }

    #[test]
    fn default_cases_cover_several_sizes() {
        let cases = CompareProtocol::default().cases();
        assert!(cases.len() >= 4);
        assert!(cases.windows(2).all(|w| w[0].spec.n_cells < w[1].spec.n_cells));
        assert!(cases.iter().all(|c| c.nclust == 5));
    }

    #[test]
    fn rust_timing_is_positive_on_a_tiny_atlas() {
        let spec = SimulateSpec {
            n_cells: 48,
            n_pcs: 4,
            n_batches: 2,
            n_types: 3,
            batch_strength: 1.5,
            seed: 3,
        };
        let atlas = simulate_atlas(&spec);
        let cfg = HarmonyConfig {
            nclust: Some(3),
            max_iter: 2,
            max_iter_cluster: 2,
            early_stop: false,
            seed: 3,
            ..HarmonyConfig::default()
        };
        let stats = time_harmorust(&atlas, &cfg, 0, 2).unwrap();
        assert_eq!(stats.times_ms.len(), 2);
        assert!(stats.median_ms > 0.0);
        assert!(stats.n_iter >= 1);
    }

    #[test]
    fn original_r_script_is_embedded() {
        assert!(ORIGINAL_R.contains("RunHarmony"));
        assert!(ORIGINAL_R.contains("proc.time"));
    }

    #[test]
    fn original_harmony_runner_when_r_package_is_installed() {
        if r_user_lib().is_none() {
            eprintln!("skipping: CRAN harmony is not installed");
            return;
        }
        let protocol = CompareProtocol {
            cell_counts: vec![36],
            n_pcs: 4,
            n_batches: 2,
            n_types: 3,
            warmup: 0,
            repeats: 1,
            max_iter: 2,
            max_iter_cluster: 2,
            ncores: 1,
            ..CompareProtocol::default()
        };
        let atlas = simulate_atlas(&protocol.cases()[0].spec);
        let workdir =
            std::env::temp_dir().join(format!("harmorust-orig-test-{}", std::process::id()));
        let (stats, version) = time_original_harmony(
            &atlas,
            &protocol,
            3,
            std::path::Path::new("Rscript"),
            &workdir,
        )
        .expect("original Harmony should run when the R package is installed");
        let _ = fs::remove_dir_all(&workdir);
        assert!(!version.is_empty());
        assert_eq!(stats.times_ms.len(), 1);
        assert!(stats.median_ms > 0.0);
    }
}
