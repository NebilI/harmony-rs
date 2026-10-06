use crate::error::{HarmonyError, Result};
use ndarray::Array2;
use std::collections::BTreeMap;

/// One-hot (and stacked multi-covariate) batch design.
#[derive(Debug, Clone)]
pub struct Design {
    /// B x N, rows are factor levels across all covariates.
    pub phi: Array2<f64>,
    /// Number of levels per covariate.
    pub levels_per_covariate: Vec<usize>,
    /// Cells belonging to each level (row of phi).
    pub index: Vec<Vec<usize>>,
    pub n_cells: usize,
    pub n_levels: usize,
    pub n_covariates: usize,
    pub pr_b: Vec<f64>,
    pub batch_sizes: Vec<f64>,
    pub level_names: Vec<String>,
}

pub fn encode_labels(labels: &[impl AsRef<str>]) -> Result<Design> {
    encode_covariates(&[labels.iter().map(|s| s.as_ref().to_string()).collect()])
}

pub fn encode_covariates(covariates: &[Vec<String>]) -> Result<Design> {
    if covariates.is_empty() {
        return Err(HarmonyError::msg("at least one batch covariate is required"));
    }
    let n = covariates[0].len();
    if n == 0 {
        return Err(HarmonyError::msg("no cells in the label vector"));
    }
    for cov in covariates {
        if cov.len() != n {
            return Err(HarmonyError::msg(
                "all covariates must have one label per cell",
            ));
        }
    }

    let mut level_names = Vec::new();
    let mut levels_per_covariate = Vec::new();
    let mut cell_codes: Vec<Vec<usize>> = Vec::new();

    for (ci, cov) in covariates.iter().enumerate() {
        let mut map: BTreeMap<String, usize> = BTreeMap::new();
        let mut codes = Vec::with_capacity(n);
        for label in cov {
            let next = map.len();
            let code = *map.entry(label.clone()).or_insert(next);
            codes.push(code);
        }
        if map.len() < 2 {
            return Err(HarmonyError::msg(format!(
                "covariate {ci} has fewer than 2 levels"
            )));
        }
        for name in map.keys() {
            level_names.push(format!("cov{ci}:{name}"));
        }
        levels_per_covariate.push(map.len());
        cell_codes.push(codes);
    }

    let n_levels: usize = levels_per_covariate.iter().sum();
    let mut phi = Array2::<f64>::zeros((n_levels, n));
    let mut index = vec![Vec::new(); n_levels];
    let mut offset = 0usize;
    for (codes, nlev) in cell_codes.iter().zip(levels_per_covariate.iter()) {
        for (cell, &code) in codes.iter().enumerate() {
            let row = offset + code;
            phi[[row, cell]] = 1.0;
            index[row].push(cell);
        }
        offset += nlev;
    }

    let batch_sizes: Vec<f64> = (0..n_levels)
        .map(|b| phi.row(b).sum())
        .collect();
    let pr_b: Vec<f64> = batch_sizes.iter().map(|s| s / n as f64).collect();

    Ok(Design {
        phi,
        levels_per_covariate,
        index,
        n_cells: n,
        n_levels,
        n_covariates: covariates.len(),
        pr_b,
        batch_sizes,
        level_names,
    })
}

/// Expand a per-covariate theta to one value per level, then scale by batch size.
pub fn expand_theta(theta_per_cov: &[f64], design: &Design, nclust: usize, tau: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(design.n_levels);
    let mut offset = 0usize;
    for (ci, &nlev) in design.levels_per_covariate.iter().enumerate() {
        let theta = theta_per_cov.get(ci).copied().unwrap_or(2.0);
        for b in 0..nlev {
            let mut t = theta;
            if tau > 0.0 {
                let nb = design.batch_sizes[offset + b];
                let scale = 1.0 - (-(nb / (nclust as f64 * tau)).powi(2)).exp();
                t *= scale;
            }
            out.push(t);
        }
        offset += nlev;
    }
    out
}

/// Expand a per-covariate lambda to `[0, λ, λ, …]` (intercept then one per level).
pub fn expand_lambda(lambda_per_cov: &[f64], design: &Design) -> Vec<f64> {
    let mut out = vec![0.0];
    let mut offset = 0usize;
    for (ci, &nlev) in design.levels_per_covariate.iter().enumerate() {
        let lam = lambda_per_cov.get(ci).copied().unwrap_or(1.0);
        for _ in 0..nlev {
            out.push(lam);
        }
        offset += nlev;
    }
    let _ = offset;
    out
}
