use serde::{Deserialize, Serialize};

/// Harmony hyperparameters, matching the R `RunHarmony` / `harmony_options` API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HarmonyConfig {
    /// Diversity penalty per covariate. Default 2. `0` does not encourage mixing.
    pub theta: f64,
    /// Soft k-means width. Default 0.1.
    pub sigma: f64,
    /// Ridge penalty. `None` enables Harmony2 automatic lambda (`alpha * E`).
    pub lambda: Option<f64>,
    /// Number of soft clusters. `None` uses Harmony's `min(round(N / 30), 100)`.
    pub nclust: Option<usize>,
    /// Outer Harmony rounds (cluster + correct).
    pub max_iter: usize,
    /// Inner clustering iterations per Harmony round.
    pub max_iter_cluster: usize,
    /// Clustering convergence tolerance (windowed).
    pub epsilon_cluster: f64,
    /// Harmony convergence tolerance.
    pub epsilon_harmony: f64,
    /// Fraction of cells updated per clustering block.
    pub block_size: f64,
    /// Harmony2 lambda scale when `lambda` is `None`.
    pub alpha: f64,
    /// Expected cells per cluster used to scale theta. `0` leaves theta unchanged.
    pub tau: f64,
    /// Skip a batch in a cluster when its soft membership is below this fraction.
    pub batch_prop_cutoff: f64,
    pub seed: u64,
    pub early_stop: bool,
    /// Full R `.Random.seed` integer vector (kind + mti + MT state). When set,
    /// the kernel starts from this stream instead of `set.seed(seed)` and
    /// writes the advanced state back on the result so the R session matches
    /// Harmony afterwards.
    #[serde(default)]
    pub rng_state: Option<Vec<i32>>,
}

impl Default for HarmonyConfig {
    fn default() -> Self {
        Self {
            theta: 2.0,
            sigma: 0.1,
            lambda: Some(1.0),
            nclust: None,
            max_iter: 10,
            max_iter_cluster: 4,
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
            block_size: 0.05,
            alpha: 0.2,
            tau: 0.0,
            batch_prop_cutoff: 1e-5,
            seed: 1,
            early_stop: true,
            rng_state: None,
        }
    }
}

/// R's `round()`: IEC 60559 round-half-to-even (banker's rounding).
pub fn r_round_half_even(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let ax = x.abs();
    let ip = ax.trunc();
    let frac = ax - ip;
    let rounded = if frac < 0.5 {
        ip
    } else if frac > 0.5 {
        ip + 1.0
    } else if (ip as i64) % 2 == 0 {
        ip
    } else {
        ip + 1.0
    };
    sign * rounded
}

impl HarmonyConfig {
    /// Harmony: `nclust <- min(round(N / 30), 100)` with R's half-to-even
    /// `round` and no lower clamp of 2. A computed 0 is raised to 1 so the
    /// kernel still has a cluster (Harmony would not run on that N anyway).
    pub fn resolved_nclust(&self, n_cells: usize) -> usize {
        self.nclust.unwrap_or_else(|| {
            let k = r_round_half_even(n_cells as f64 / 30.0) as i64;
            k.clamp(1, 100) as usize
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r_round_half_even_matches_r() {
        assert_eq!(r_round_half_even(2.5), 2.0);
        assert_eq!(r_round_half_even(3.5), 4.0);
        assert_eq!(r_round_half_even(16.666666666666668), 17.0);
        assert_eq!(r_round_half_even(0.2), 0.0);
        assert_eq!(r_round_half_even(-2.5), -2.0);
    }

    #[test]
    fn auto_nclust_matches_harmony_round() {
        let cfg = HarmonyConfig::default();
        assert_eq!(cfg.resolved_nclust(500), 17);
        // 75/30 = 2.5 → R half-even → 2 (Rust f64::round would give 3).
        assert_eq!(cfg.resolved_nclust(75), 2);
        assert_eq!(cfg.resolved_nclust(45), 2);
        // 15/30 = 0.5 → half-even → 0, raised to 1 (no clamp-2).
        assert_eq!(cfg.resolved_nclust(15), 1);
        assert_eq!(cfg.resolved_nclust(8000), 100);
    }
}
