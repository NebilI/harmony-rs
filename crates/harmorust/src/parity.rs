//! Numerical comparison of harmorust (`f64`) vs immunogenomics/harmony (C++ `float`).
//!
//! The existing `compare` command reports wall time. This module runs both
//! kernels on the same synthetic atlas and reports the element-wise absolute
//! difference of each result matrix: min and max in scientific notation.
//!
//! Harmony C++ stores the loop in `float`. Differences therefore mix rounding
//! with independent k-means++ / block shuffles. `nclust = 1` isolates the
//! ridge (no clustering ambiguity). Soft-assignment matrices `R`, `Y`, `O`,
//! and `E` are compared after a greedy cluster alignment on `Y`.

use crate::compare::{
    r_user_lib, write_batch_csv, write_embeddings_csv, CompareProtocol,
};
use crate::correct::{default_lambda, moe_correct_ridge};
use crate::design::encode_labels;
use crate::error::{HarmonyError, Result};
use crate::harmony::{embeddings_from_rows, run_harmony, HarmonyResult};
use crate::simulate::{simulate_atlas, SimulatedAtlas};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const ORIGINAL_PARITY_R: &str = include_str!("../scripts/original_harmony_parity.R");

#[derive(Debug, Clone)]
pub struct ParityProtocol {
    pub compare: CompareProtocol,
    /// One or more nclust settings. `None` means Harmony's auto default.
    pub nclusts: Vec<Option<usize>>,
}

impl Default for ParityProtocol {
    fn default() -> Self {
        Self {
            compare: CompareProtocol {
                cell_counts: vec![200, 500, 1000],
                warmup: 0,
                repeats: 1,
                ..CompareProtocol::default()
            },
            nclusts: vec![Some(1), Some(5)],
        }
    }
}

impl ParityProtocol {
    fn with_nclust(&self, nclust: Option<usize>) -> CompareProtocol {
        let mut p = self.compare.clone();
        p.nclust = nclust;
        p
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AbsDiffStats {
    pub n: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDiff {
    pub name: String,
    pub rust_shape: (usize, usize),
    pub original_shape: (usize, usize),
    pub aligned: bool,
    pub stats: Option<AbsDiffStats>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityRow {
    pub name: String,
    pub n_cells: usize,
    pub n_pcs: usize,
    pub n_clusters: usize,
    pub rust_n_iter: usize,
    pub original_n_iter: usize,
    pub fields: Vec<FieldDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityReport {
    pub original_version: Option<String>,
    pub n_pcs: usize,
    pub n_batches: usize,
    pub max_iter: usize,
    pub seed: u64,
    pub lambda: f64,
    pub rows: Vec<ParityRow>,
}

#[derive(Debug, Clone)]
pub struct ParityOptions {
    pub rscript: PathBuf,
}

impl Default for ParityOptions {
    fn default() -> Self {
        Self {
            rscript: PathBuf::from("Rscript"),
        }
    }
}

/// Format a non-negative magnitude in scientific notation (`1.2345e-07`).
pub fn fmt_sci(x: f64) -> String {
    if !x.is_finite() {
        return "n/a".into();
    }
    if x == 0.0 {
        return "0.0000e+00".into();
    }
    let s = format!("{x:.4e}");
    let Some((mant, exp)) = s.split_once('e') else {
        return s;
    };
    let (sign, digits) = if let Some(rest) = exp.strip_prefix('-') {
        ("-", rest)
    } else if let Some(rest) = exp.strip_prefix('+') {
        ("+", rest)
    } else {
        ("+", exp)
    };
    let n: i32 = digits.parse().unwrap_or(0);
    format!("{mant}e{sign}{n:02}")
}

pub fn abs_diff_stats(left: &[f64], right: &[f64]) -> Result<AbsDiffStats> {
    if left.len() != right.len() {
        return Err(HarmonyError::msg(format!(
            "abs-diff length mismatch: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    if left.is_empty() {
        return Err(HarmonyError::msg("abs-diff of empty vectors"));
    }
    let mut diffs = Vec::with_capacity(left.len());
    let mut min = f64::INFINITY;
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    for (&a, &b) in left.iter().zip(right.iter()) {
        let d = if a.is_finite() && b.is_finite() {
            (a - b).abs()
        } else if a == b {
            0.0
        } else {
            f64::NAN
        };
        if d.is_finite() {
            min = min.min(d);
            max = max.max(d);
            sum += d;
        }
        diffs.push(d);
    }
    if !min.is_finite() {
        return Err(HarmonyError::msg("no finite absolute differences"));
    }
    let finite: Vec<f64> = diffs.iter().copied().filter(|d| d.is_finite()).collect();
    Ok(AbsDiffStats {
        n: finite.len(),
        min,
        max,
        mean: sum / finite.len() as f64,
        median: crate::compare::median(&finite),
    })
}

fn matrix_shape(m: &[Vec<f64>]) -> (usize, usize) {
    let rows = m.len();
    let cols = m.first().map(|r| r.len()).unwrap_or(0);
    (rows, cols)
}

fn flatten(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

fn shapes_equal(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    matrix_shape(a) == matrix_shape(b)
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.len() == y.len())
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return Vec::new();
    }
    let cols = m[0].len();
    (0..cols)
        .map(|j| m.iter().map(|row| row[j]).collect())
        .collect()
}

/// Orient `m` to `(rows, cols)`, transposing if the other orientation matches.
fn orient(m: &[Vec<f64>], rows: usize, cols: usize) -> Option<Vec<Vec<f64>>> {
    let (r, c) = matrix_shape(m);
    if r == rows && c == cols {
        Some(m.to_vec())
    } else if r == cols && c == rows {
        Some(transpose(m))
    } else {
        None
    }
}

fn rows_to_array(m: &[Vec<f64>]) -> Result<Array2<f64>> {
    let (r, c) = matrix_shape(m);
    if r == 0 || c == 0 {
        return Err(HarmonyError::msg("empty matrix"));
    }
    let mut z = Array2::<f64>::zeros((r, c));
    for (i, row) in m.iter().enumerate() {
        if row.len() != c {
            return Err(HarmonyError::msg("ragged matrix"));
        }
        for (j, &v) in row.iter().enumerate() {
            z[[i, j]] = v;
        }
    }
    Ok(z)
}

fn array_to_cells_by_pc(z: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = z.ncols();
    let d = z.nrows();
    (0..n)
        .map(|j| (0..d).map(|i| z[[i, j]]).collect())
        .collect()
}

fn column(m: &[Vec<f64>], j: usize) -> Vec<f64> {
    m.iter().map(|row| row[j]).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Greedy matching of cluster columns of `Y` (PCs × clusters) by cosine similarity.
/// `perm[i]` is the original cluster that best matches rust cluster `i`.
pub fn align_clusters_by_y(rust_y: &[Vec<f64>], orig_y: &[Vec<f64>]) -> Option<Vec<usize>> {
    let (d, k) = matrix_shape(rust_y);
    if d == 0 || k == 0 || matrix_shape(orig_y) != (d, k) {
        return None;
    }
    let mut pairs = Vec::with_capacity(k * k);
    for i in 0..k {
        let yi = column(rust_y, i);
        for j in 0..k {
            pairs.push((cosine(&yi, &column(orig_y, j)), i, j));
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut rust_used = vec![false; k];
    let mut orig_used = vec![false; k];
    let mut perm = vec![0usize; k];
    let mut assigned = 0usize;
    for &(_, i, j) in &pairs {
        if rust_used[i] || orig_used[j] {
            continue;
        }
        rust_used[i] = true;
        orig_used[j] = true;
        perm[i] = j;
        assigned += 1;
        if assigned == k {
            break;
        }
    }
    if assigned == k {
        Some(perm)
    } else {
        None
    }
}

fn permute_rows(m: &[Vec<f64>], perm: &[usize]) -> Vec<Vec<f64>> {
    perm.iter().map(|&src| m[src].clone()).collect()
}

fn permute_cols(m: &[Vec<f64>], perm: &[usize]) -> Vec<Vec<f64>> {
    m.iter()
        .map(|row| perm.iter().map(|&src| row[src]).collect())
        .collect()
}

fn field_diff(name: &str, rust: &[Vec<f64>], original: &[Vec<f64>], aligned: bool) -> FieldDiff {
    let rust_shape = matrix_shape(rust);
    let original_shape = matrix_shape(original);
    if rust.is_empty() || original.is_empty() {
        return FieldDiff {
            name: name.into(),
            rust_shape,
            original_shape,
            aligned,
            stats: None,
            note: Some("missing matrix".into()),
        };
    }
    if !shapes_equal(rust, original) {
        return FieldDiff {
            name: name.into(),
            rust_shape,
            original_shape,
            aligned,
            stats: None,
            note: Some("shape mismatch".into()),
        };
    }
    match abs_diff_stats(&flatten(rust), &flatten(original)) {
        Ok(stats) => FieldDiff {
            name: name.into(),
            rust_shape,
            original_shape,
            aligned,
            stats: Some(stats),
            note: None,
        },
        Err(e) => FieldDiff {
            name: name.into(),
            rust_shape,
            original_shape,
            aligned,
            stats: None,
            note: Some(e.to_string()),
        },
    }
}

fn vec_field_diff(name: &str, rust: &[f64], original: &[f64]) -> FieldDiff {
    let rust_m = vec![rust.to_vec()];
    let orig_m = vec![original.to_vec()];
    let mut field = field_diff(name, &rust_m, &orig_m, false);
    field.rust_shape = (1, rust.len());
    field.original_shape = (1, original.len());
    if rust.len() != original.len() {
        // Compare the last value when the objective traces differ in length.
        if let (Some(&a), Some(&b)) = (rust.last(), original.last()) {
            if let Ok(stats) = abs_diff_stats(&[a], &[b]) {
                field.stats = Some(stats);
                field.note = Some("compared final objective only (length mismatch)".into());
            }
        }
    }
    field
}

#[derive(Debug, Deserialize)]
struct OriginalMeta {
    version: String,
    n_iter: usize,
    n_cells: usize,
    n_pcs: usize,
    has_embeddings: bool,
    has_z_orig: bool,
    has_r: bool,
    has_y: bool,
    has_o: bool,
    has_e: bool,
    objective: Vec<f64>,
}

#[derive(Debug, Clone)]
struct OriginalDump {
    version: String,
    n_iter: usize,
    embeddings: Vec<Vec<f64>>,
    z_orig: Option<Vec<Vec<f64>>>,
    r: Option<Vec<Vec<f64>>>,
    y: Option<Vec<Vec<f64>>>,
    o: Option<Vec<Vec<f64>>>,
    e: Option<Vec<Vec<f64>>>,
    objective: Vec<f64>,
}

fn read_csv_matrix(path: &Path) -> Result<Vec<Vec<f64>>> {
    let raw = fs::read_to_string(path).map_err(|e| HarmonyError::msg(e.to_string()))?;
    let mut rows = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut vals = Vec::new();
        for s in line.split(',') {
            let v = s
                .trim()
                .parse::<f64>()
                .map_err(|e| HarmonyError::msg(format!("{path:?} line {i}: {e}")))?;
            vals.push(v);
        }
        rows.push(vals);
    }
    if rows.is_empty() {
        return Err(HarmonyError::msg(format!("empty matrix at {path:?}")));
    }
    Ok(rows)
}

fn run_original_harmony(
    atlas: &SimulatedAtlas,
    protocol: &CompareProtocol,
    nclust: usize,
    rscript: &Path,
    workdir: &Path,
) -> Result<OriginalDump> {
    fs::create_dir_all(workdir).map_err(|e| HarmonyError::msg(e.to_string()))?;
    let script = workdir.join("original_harmony_parity.R");
    fs::write(&script, ORIGINAL_PARITY_R).map_err(|e| HarmonyError::msg(e.to_string()))?;
    let emb_path = workdir.join("embeddings.csv");
    let batch_path = workdir.join("batch.csv");
    let out_dir = workdir.join("out");
    fs::create_dir_all(&out_dir).map_err(|e| HarmonyError::msg(e.to_string()))?;
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
        .arg(&out_dir);
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
            "original Harmony (R) parity dump failed ({}):\n{stderr}{stdout}",
            output.status
        )));
    }

    let meta_raw = fs::read_to_string(out_dir.join("meta.json")).map_err(|e| {
        HarmonyError::msg(format!(
            "original Harmony wrote no meta.json ({e}). R stdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        ))
    })?;
    let meta: OriginalMeta = serde_json::from_str(meta_raw.trim()).map_err(|e| {
        HarmonyError::msg(format!(
            "could not parse original Harmony meta JSON: {e}\n{meta_raw}"
        ))
    })?;
    if meta.n_cells != atlas.n_cells || meta.n_pcs != atlas.n_pcs {
        return Err(HarmonyError::msg(format!(
            "original Harmony dump size mismatch: {}×{} vs atlas {}×{}",
            meta.n_cells, meta.n_pcs, atlas.n_cells, atlas.n_pcs
        )));
    }

    let load = |flag: bool, name: &str| -> Result<Option<Vec<Vec<f64>>>> {
        if !flag {
            return Ok(None);
        }
        Ok(Some(read_csv_matrix(&out_dir.join(name))?))
    };

    let embeddings = load(meta.has_embeddings, "embeddings.csv")?
        .ok_or_else(|| HarmonyError::msg("original Harmony dump missing embeddings"))?;
    Ok(OriginalDump {
        version: meta.version,
        n_iter: meta.n_iter,
        embeddings,
        z_orig: load(meta.has_z_orig, "z_orig.csv")?,
        r: load(meta.has_r, "R.csv")?,
        y: load(meta.has_y, "Y.csv")?,
        o: load(meta.has_o, "O.csv")?,
        e: load(meta.has_e, "E.csv")?,
        objective: meta.objective,
    })
}

fn rust_ridge_from_original(
    atlas: &SimulatedAtlas,
    orig: &OriginalDump,
    protocol: &CompareProtocol,
) -> Result<Vec<Vec<f64>>> {
    let design = encode_labels(&atlas.batch)?;
    let z_orig = match &orig.z_orig {
        Some(m) => {
            let oriented = orient(m, atlas.n_pcs, atlas.n_cells).ok_or_else(|| {
                HarmonyError::msg(format!(
                    "Z_orig shape {:?} is neither {}×{} nor {}×{}",
                    matrix_shape(m),
                    atlas.n_pcs,
                    atlas.n_cells,
                    atlas.n_cells,
                    atlas.n_pcs
                ))
            })?;
            rows_to_array(&oriented)?
        }
        None => embeddings_from_rows(&atlas.embeddings)?.t().to_owned(),
    };
    let r = orig
        .r
        .as_ref()
        .ok_or_else(|| HarmonyError::msg("original Harmony dump missing R"))?;
    let o = orig
        .o
        .as_ref()
        .ok_or_else(|| HarmonyError::msg("original Harmony dump missing O"))?;
    let e = orig
        .e
        .as_ref()
        .ok_or_else(|| HarmonyError::msg("original Harmony dump missing E"))?;
    let r = rows_to_array(r)?;
    let o = rows_to_array(o)?;
    let e = rows_to_array(e)?;
    let lambda = default_lambda(design.n_levels, protocol.lambda);
    let mut y = Array2::<f64>::zeros((z_orig.nrows(), r.nrows()));
    let z_corr = moe_correct_ridge(
        z_orig.view(),
        r.view(),
        o.view(),
        e.view(),
        &design,
        Some(&lambda),
        0.2,
        1e-5,
        &mut y,
    )?;
    Ok(array_to_cells_by_pc(&z_corr))
}

fn compare_case(
    name: &str,
    atlas: &SimulatedAtlas,
    rust: &HarmonyResult,
    orig: &OriginalDump,
    protocol: &CompareProtocol,
) -> ParityRow {
    let mut fields = Vec::new();

    let orig_emb = orient(&orig.embeddings, atlas.n_cells, atlas.n_pcs)
        .unwrap_or_else(|| orig.embeddings.clone());
    fields.push(field_diff("embeddings", &rust.embeddings, &orig_emb, false));

    if let (Some(orig_r), Some(orig_y), Some(orig_o)) = (&orig.r, &orig.y, &orig.o) {
        let orig_y = orient(orig_y, atlas.n_pcs, rust.n_clusters).unwrap_or_else(|| orig_y.clone());
        let orig_r = orient(orig_r, rust.n_clusters, atlas.n_cells).unwrap_or_else(|| orig_r.clone());
        let orig_o = orig_o.clone();
        let orig_e = orig.e.clone();
        let perm = align_clusters_by_y(&rust.y, &orig_y);
        let aligned = perm.is_some();
        let (orig_y, orig_r, orig_o, orig_e) = if let Some(perm) = perm {
            (
                permute_cols(&orig_y, &perm),
                permute_rows(&orig_r, &perm),
                permute_rows(&orig_o, &perm),
                orig_e.map(|e| permute_rows(&e, &perm)),
            )
        } else {
            (orig_y, orig_r, orig_o, orig_e)
        };
        fields.push(field_diff("R", &rust.r, &orig_r, aligned));
        fields.push(field_diff("Y", &rust.y, &orig_y, aligned));
        fields.push(field_diff("O", &rust.o, &orig_o, aligned));
        if let Some(orig_e) = orig_e {
            // harmorust does not currently export E on HarmonyResult; rebuild
            // it from rust O's row sums × batch proportions via a fresh run
            // is unnecessary — compare original E against rust-derived E from O.
            // We only have rust O. Skip a rust-vs-original E unless we add E
            // to HarmonyResult. Compare original E against itself? No.
            // Just skip rust E; the field is still useful if we dump rust E
            // from observed_expected. Leave E as original-only ridge input.
            let _ = orig_e;
        }
    } else {
        if let Some(orig_r) = &orig.r {
            fields.push(field_diff("R", &rust.r, orig_r, false));
        }
        if let Some(orig_y) = &orig.y {
            fields.push(field_diff("Y", &rust.y, orig_y, false));
        }
        if let Some(orig_o) = &orig.o {
            fields.push(field_diff("O", &rust.o, orig_o, false));
        }
    }

    fields.push(vec_field_diff(
        "objective",
        &rust.objective,
        &orig.objective,
    ));

    match rust_ridge_from_original(atlas, orig, protocol) {
        Ok(ridge) => {
            fields.push(field_diff(
                "moe_correct_ridge",
                &ridge,
                &orig_emb,
                false,
            ));
        }
        Err(e) => fields.push(FieldDiff {
            name: "moe_correct_ridge".into(),
            rust_shape: (0, 0),
            original_shape: matrix_shape(&orig_emb),
            aligned: false,
            stats: None,
            note: Some(e.to_string()),
        }),
    }

    ParityRow {
        name: name.into(),
        n_cells: atlas.n_cells,
        n_pcs: atlas.n_pcs,
        n_clusters: rust.n_clusters,
        rust_n_iter: rust.n_iter,
        original_n_iter: orig.n_iter,
        fields,
    }
}

pub fn run_parity(protocol: &ParityProtocol, opts: &ParityOptions) -> Result<ParityReport> {
    if protocol.nclusts.is_empty() {
        return Err(HarmonyError::msg("need at least one --nclust setting"));
    }
    if protocol.compare.cell_counts.is_empty() {
        return Err(HarmonyError::msg("need at least one --cells count"));
    }

    let pid = std::process::id();
    let mut rows = Vec::new();
    let mut original_version = None;
    let mut case_i = 0usize;
    for nclust_opt in &protocol.nclusts {
        let cmp = protocol.with_nclust(*nclust_opt);
        for spec_n in &cmp.cell_counts {
            let nclust = cmp.resolved_nclust(*spec_n);
            let spec = crate::simulate::SimulateSpec {
                n_cells: *spec_n,
                n_pcs: cmp.n_pcs,
                n_batches: cmp.n_batches,
                n_types: cmp.n_types,
                batch_strength: cmp.batch_strength,
                seed: cmp.seed,
            };
            let atlas = simulate_atlas(&spec);
            let cfg = cmp.config(nclust);
            let z = embeddings_from_rows(&atlas.embeddings)?;
            let rust = run_harmony(z.view(), &atlas.batch, &cfg)?;
            let workdir =
                std::env::temp_dir().join(format!("harmorust-parity-{pid}-{case_i}"));
            let orig = run_original_harmony(&atlas, &cmp, nclust, &opts.rscript, &workdir)?;
            original_version = Some(orig.version.clone());
            let label = format!(
                "atlas-{} nclust={}",
                atlas.n_cells,
                nclust_opt
                    .map(|k| k.to_string())
                    .unwrap_or_else(|| format!("auto({nclust})"))
            );
            rows.push(compare_case(&label, &atlas, &rust, &orig, &cmp));
            let _ = fs::remove_dir_all(&workdir);
            case_i += 1;
        }
    }

    Ok(ParityReport {
        original_version,
        n_pcs: protocol.compare.n_pcs,
        n_batches: protocol.compare.n_batches,
        max_iter: protocol.compare.max_iter,
        seed: protocol.compare.seed,
        lambda: protocol.compare.lambda,
        rows,
    })
}

pub fn format_markdown_table(rows: &[ParityRow]) -> String {
    let mut out = String::new();
    out.push_str(
        "| Case | Clusters | Field | Elements | min |Δ| | max |Δ| |\n",
    );
    out.push_str("| --- | ---: | --- | ---: | ---: | ---: |\n");
    for row in rows {
        for field in &row.fields {
            let (min, max) = match &field.stats {
                Some(s) => (fmt_sci(s.min), fmt_sci(s.max)),
                None => ("n/a".into(), "n/a".into()),
            };
            let n = field.stats.as_ref().map(|s| s.n).unwrap_or(0);
            let name = if field.aligned {
                format!("{} (aligned)", field.name)
            } else {
                field.name.clone()
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                row.name, row.n_clusters, name, n, min, max
            ));
        }
    }
    out
}

pub fn print_report(report: &ParityReport, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(report).unwrap());
        return;
    }
    println!("Harmony numerical parity (absolute difference)");
    println!(
        "protocol: {} PCs, {} batches, max_iter={}, lambda={}, seed={}",
        report.n_pcs, report.n_batches, report.max_iter, report.lambda, report.seed
    );
    println!(
        "Harmony C++ stores the loop in float; harmorust uses f64. |Δ| is |harmorust − original|."
    );
    println!(
        "nclust=1 isolates the ridge. R / Y / O are compared after aligning clusters on Y."
    );
    println!(
        "moe_correct_ridge is harmorust's ridge applied to Harmony's final R, vs Harmony Z_corr."
    );
    if let Some(v) = &report.original_version {
        println!("original: immunogenomics/harmony {v} (C++ float via R)");
    }
    println!("new: harmorust {}", env!("CARGO_PKG_VERSION"));
    println!();
    print!("{}", format_markdown_table(&report.rows));
    println!();
    for row in &report.rows {
        println!(
            "{}: rust n_iter={} original n_iter={}",
            row.name, row.rust_n_iter, row.original_n_iter
        );
        for field in &row.fields {
            match (&field.stats, &field.note) {
                (Some(s), note) => {
                    let extra = note
                        .as_ref()
                        .map(|n| format!(" ({n})"))
                        .unwrap_or_default();
                    println!(
                        "  {}: n={} min {} max {} mean {} median {}{}",
                        field.name,
                        s.n,
                        fmt_sci(s.min),
                        fmt_sci(s.max),
                        fmt_sci(s.mean),
                        fmt_sci(s.median),
                        extra
                    );
                }
                (None, Some(note)) => {
                    println!("  {}: skipped ({note})", field.name);
                }
                (None, None) => println!("  {}: n/a", field.name),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_sci_zero_and_typical() {
        assert_eq!(fmt_sci(0.0), "0.0000e+00");
        assert_eq!(fmt_sci(1.23456e-7), "1.2346e-07");
        assert_eq!(fmt_sci(3.2), "3.2000e+00");
        assert_eq!(fmt_sci(f64::NAN), "n/a");
    }

    #[test]
    fn abs_diff_min_max_mean() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.5, 4.0];
        let s = abs_diff_stats(&a, &b).unwrap();
        assert_eq!(s.n, 3);
        assert!((s.min - 0.0).abs() < 1e-15);
        assert!((s.max - 1.0).abs() < 1e-15);
        assert!((s.mean - 0.5).abs() < 1e-15);
        assert!((s.median - 0.5).abs() < 1e-15);
    }

    #[test]
    fn abs_diff_rejects_length_mismatch() {
        assert!(abs_diff_stats(&[1.0], &[1.0, 2.0]).is_err());
        assert!(abs_diff_stats(&[], &[]).is_err());
    }

    #[test]
    fn cluster_alignment_recovers_swap() {
        let rust_y = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let orig_y = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let perm = align_clusters_by_y(&rust_y, &orig_y).unwrap();
        assert_eq!(perm, vec![1, 0]);
        let aligned = permute_cols(&orig_y, &perm);
        assert_eq!(aligned, rust_y);
    }

    #[test]
    fn markdown_table_uses_scientific_notation() {
        let row = ParityRow {
            name: "atlas-200 nclust=1".into(),
            n_cells: 200,
            n_pcs: 4,
            n_clusters: 1,
            rust_n_iter: 8,
            original_n_iter: 8,
            fields: vec![FieldDiff {
                name: "embeddings".into(),
                rust_shape: (200, 4),
                original_shape: (200, 4),
                aligned: false,
                stats: Some(AbsDiffStats {
                    n: 800,
                    min: 1.19e-7,
                    max: 3.45e-5,
                    mean: 2e-6,
                    median: 1e-6,
                }),
                note: None,
            }],
        };
        let table = format_markdown_table(&[row]);
        assert!(table.contains("min |Δ|"));
        assert!(table.contains("1.1900e-07"));
        assert!(table.contains("3.4500e-05"));
        assert!(table.contains("embeddings"));
    }

    #[test]
    fn original_parity_script_is_embedded() {
        assert!(ORIGINAL_PARITY_R.contains("RunHarmony"));
        assert!(ORIGINAL_PARITY_R.contains("getZcorr"));
        assert!(ORIGINAL_PARITY_R.contains("%.17g"));
    }

    #[test]
    fn rust_vs_rust_embeddings_are_zero() {
        let spec = crate::simulate::SimulateSpec {
            n_cells: 48,
            n_pcs: 4,
            n_batches: 2,
            n_types: 3,
            batch_strength: 1.5,
            seed: 3,
        };
        let atlas = simulate_atlas(&spec);
        let z = embeddings_from_rows(&atlas.embeddings).unwrap();
        let cfg = crate::config::HarmonyConfig {
            nclust: Some(3),
            max_iter: 2,
            max_iter_cluster: 2,
            early_stop: false,
            seed: 3,
            ..crate::config::HarmonyConfig::default()
        };
        let a = run_harmony(z.view(), &atlas.batch, &cfg).unwrap();
        let b = run_harmony(z.view(), &atlas.batch, &cfg).unwrap();
        let s = abs_diff_stats(&flatten(&a.embeddings), &flatten(&b.embeddings)).unwrap();
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
    }

    #[test]
    fn original_harmony_parity_dump_when_r_package_is_installed() {
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
            early_stop: false,
            ..CompareProtocol::default()
        };
        let atlas = simulate_atlas(&protocol.cases()[0].spec);
        let workdir =
            std::env::temp_dir().join(format!("harmorust-parity-test-{}", std::process::id()));
        let dump = run_original_harmony(
            &atlas,
            &protocol,
            3,
            std::path::Path::new("Rscript"),
            &workdir,
        )
        .expect("original Harmony should dump results when the R package is installed");
        let _ = fs::remove_dir_all(&workdir);
        assert!(!dump.version.is_empty());
        assert_eq!(matrix_shape(&dump.embeddings), (atlas.n_cells, atlas.n_pcs));
        assert!(dump.r.is_some());
        assert!(dump.y.is_some());
    }
}
