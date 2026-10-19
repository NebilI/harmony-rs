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
    /// Number of soft clusters. `None` uses `min(round(N / 30), 100)`.
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
        }
    }
}

impl HarmonyConfig {
    pub fn resolved_nclust(&self, n_cells: usize) -> usize {
        self.nclust
            .unwrap_or_else(|| ((n_cells as f64 / 30.0).round() as usize).clamp(2, 100))
    }
}
