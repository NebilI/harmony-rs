use crate::cluster::{init_r, kmeans_cosine, objective, observed_expected, update_r};
use crate::config::HarmonyConfig;
use crate::correct::{default_lambda, moe_correct_ridge};
use crate::design::{encode_covariates, encode_labels, expand_lambda, expand_theta, Design};
use crate::error::{HarmonyError, Result};
use crate::linalg::l2_normalize_columns;
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyResult {
    /// Corrected embeddings, cells × PCs (same orientation Harmony-R returns).
    pub embeddings: Vec<Vec<f64>>,
    /// Soft cluster assignments R, clusters × cells.
    pub r: Vec<Vec<f64>>,
    /// Cluster centroids Y, PCs × clusters, L2-normalised.
    pub y: Vec<Vec<f64>>,
    /// Observed cluster×level counts O.
    pub o: Vec<Vec<f64>>,
    pub n_iter: usize,
    pub converged: bool,
    pub n_clusters: usize,
    pub objective: Vec<f64>,
    pub kmeans_rounds: Vec<usize>,
}

pub struct HarmonyState {
    pub z_orig: Array2<f64>,
    pub z_corr: Array2<f64>,
    pub y: Array2<f64>,
    pub r: Array2<f64>,
    pub dist: Array2<f64>,
    pub o: Array2<f64>,
    pub e: Array2<f64>,
    pub design: Design,
    pub sigma: Array1<f64>,
    pub theta: Vec<f64>,
    pub lambda: Option<Vec<f64>>,
    pub config: HarmonyConfig,
    pub objective_kmeans: Vec<f64>,
    pub objective_harmony: Vec<f64>,
    pub kmeans_rounds: Vec<usize>,
}

/// Run Harmony on a cells × PCs embedding and a single batch vector.
pub fn run_harmony(
    embeddings: ArrayView2<'_, f64>,
    batch: &[impl AsRef<str>],
    config: &HarmonyConfig,
) -> Result<HarmonyResult> {
    let design = encode_labels(batch)?;
    let mut state = HarmonyState::setup(embeddings, design, config)?;
    state.run()
}

/// Run Harmony with one or more categorical covariates (stacked like Harmony-R).
pub fn run_harmony_multi(
    embeddings: ArrayView2<'_, f64>,
    covariates: &[Vec<String>],
    theta: &[f64],
    lambda: Option<&[f64]>,
    config: &HarmonyConfig,
) -> Result<HarmonyResult> {
    let design = encode_covariates(covariates)?;
    let mut cfg = config.clone();
    if let Some(first) = theta.first() {
        cfg.theta = *first;
    }
    let mut state = HarmonyState::setup(embeddings, design, &cfg)?;
    if theta.len() != state.design.n_covariates && theta.len() != 1 {
        return Err(HarmonyError::msg(
            "theta must be one value or one per covariate",
        ));
    }
    let thetas = if theta.len() == 1 {
        vec![theta[0]; state.design.n_covariates]
    } else {
        theta.to_vec()
    };
    state.theta = expand_theta(
        &thetas,
        &state.design,
        state.y.ncols(),
        cfg.tau,
    );
    if let Some(lams) = lambda {
        state.lambda = Some(expand_lambda(lams, &state.design));
    }
    state.run()
}

impl HarmonyState {
    pub fn setup(
        embeddings: ArrayView2<'_, f64>,
        design: Design,
        config: &HarmonyConfig,
    ) -> Result<Self> {
        let n = embeddings.nrows();
        let d = embeddings.ncols();
        if n != design.n_cells {
            return Err(HarmonyError::msg(format!(
                "embedding has {n} cells but labels have {}",
                design.n_cells
            )));
        }
        if n < 6 {
            return Err(HarmonyError::msg("refusing to run with fewer than 6 cells"));
        }
        if d == 0 {
            return Err(HarmonyError::msg("embedding has no principal components"));
        }

        let mut block_size = config.block_size;
        if n < 40 {
            block_size = 0.2;
        }

        let k = config.resolved_nclust(n);
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        // Internal layout is PCs × cells, matching the C++ Harmony object.
        let z_orig = embeddings.t().to_owned();
        let z_corr = l2_normalize_columns(z_orig.view());
        let y = kmeans_cosine(z_corr.view(), k, &mut rng, 10);
        let sigma = Array1::from_elem(k, config.sigma);
        let (r, dist) = init_r(y.view(), z_corr.view(), &sigma);
        let (o, e) = observed_expected(r.view(), design.phi.view(), &design.pr_b);

        let theta = expand_theta(
            &vec![config.theta; design.n_covariates],
            &design,
            k,
            config.tau,
        );
        let lambda = config.lambda.map(|l| default_lambda(design.n_levels, l));

        let mut cfg = config.clone();
        cfg.block_size = block_size;

        let mut state = Self {
            z_orig,
            z_corr,
            y,
            r,
            dist,
            o,
            e,
            design,
            sigma,
            theta,
            lambda,
            config: cfg,
            objective_kmeans: Vec::new(),
            objective_harmony: Vec::new(),
            kmeans_rounds: Vec::new(),
        };
        let obj = state.current_objective();
        state.objective_kmeans.push(obj);
        state.objective_harmony.push(obj);
        Ok(state)
    }

    fn current_objective(&self) -> f64 {
        objective(
            self.r.view(),
            self.dist.view(),
            self.o.view(),
            self.e.view(),
            self.design.phi.view(),
            &self.theta,
            &self.sigma,
        )
    }

    fn cluster_round(&mut self, rng: &mut ChaCha8Rng) {
        // After a correction, re-estimate R from the new Z_corr and current Y.
        if !self.kmeans_rounds.is_empty() {
            self.z_corr = l2_normalize_columns(self.z_corr.view());
            let (r, dist) = init_r(self.y.view(), self.z_corr.view(), &self.sigma);
            self.r = r;
            self.dist = dist;
            let (o, e) =
                observed_expected(self.r.view(), self.design.phi.view(), &self.design.pr_b);
            self.o = o;
            self.e = e;
        }

        let mut used = 0usize;
        for iter in 0..self.config.max_iter_cluster {
            used = iter + 1;
            update_r(
                &mut self.r,
                &self.dist,
                &self.design,
                &self.theta,
                &self.sigma,
                &mut self.o,
                &mut self.e,
                self.config.block_size,
                rng,
            );
            let obj = self.current_objective();
            self.objective_kmeans.push(obj);
            if iter + 1 > 3 && self.cluster_converged() {
                break;
            }
        }
        self.kmeans_rounds.push(used);
        self.objective_harmony
            .push(*self.objective_kmeans.last().unwrap());
    }

    fn cluster_converged(&self) -> bool {
        let objs = &self.objective_kmeans;
        if objs.len() < 5 {
            return false;
        }
        let window = 3;
        let mut old = 0.0;
        let mut new = 0.0;
        for i in 0..window {
            old += objs[objs.len() - 2 - i];
            new += objs[objs.len() - 1 - i];
        }
        if old.abs() < 1e-12 {
            return true;
        }
        (old - new).abs() / old.abs() < self.config.epsilon_cluster
    }

    fn harmony_converged(&self) -> bool {
        if !self.config.early_stop || self.objective_harmony.len() < 2 {
            return false;
        }
        let old = self.objective_harmony[self.objective_harmony.len() - 2];
        let new = self.objective_harmony[self.objective_harmony.len() - 1];
        if old.abs() < 1e-12 {
            return true;
        }
        (old - new) / old.abs() < self.config.epsilon_harmony
    }

    pub fn run(&mut self) -> Result<HarmonyResult> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed.wrapping_add(17));
        let mut n_iter = 0usize;
        let mut converged = false;
        for _ in 0..self.config.max_iter {
            n_iter += 1;
            self.cluster_round(&mut rng);
            self.z_corr = moe_correct_ridge(
                self.z_orig.view(),
                self.r.view(),
                self.o.view(),
                self.e.view(),
                &self.design,
                self.lambda.as_deref(),
                self.config.alpha,
                self.config.batch_prop_cutoff,
                &mut self.y,
            )?;
            if self.harmony_converged() {
                converged = true;
                break;
            }
        }

        Ok(HarmonyResult {
            embeddings: array_to_cells_by_pc(&self.z_corr),
            r: array_to_rows(&self.r),
            y: array_to_rows(&self.y),
            o: array_to_rows(&self.o),
            n_iter,
            converged,
            n_clusters: self.y.ncols(),
            objective: self.objective_harmony.clone(),
            kmeans_rounds: self.kmeans_rounds.clone(),
        })
    }
}

fn array_to_cells_by_pc(z: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = z.ncols();
    let d = z.nrows();
    (0..n)
        .map(|j| (0..d).map(|i| z[[i, j]]).collect())
        .collect()
}

fn array_to_rows(z: &Array2<f64>) -> Vec<Vec<f64>> {
    (0..z.nrows())
        .map(|i| (0..z.ncols()).map(|j| z[[i, j]]).collect())
        .collect()
}

pub fn embeddings_from_rows(rows: &[Vec<f64>]) -> Result<Array2<f64>> {
    if rows.is_empty() {
        return Err(HarmonyError::msg("empty embedding"));
    }
    let n = rows.len();
    let d = rows[0].len();
    if d == 0 {
        return Err(HarmonyError::msg("embedding has no columns"));
    }
    let mut z = Array2::<f64>::zeros((n, d));
    for (i, row) in rows.iter().enumerate() {
        if row.len() != d {
            return Err(HarmonyError::msg("ragged embedding matrix"));
        }
        for (j, &v) in row.iter().enumerate() {
            z[[i, j]] = v;
        }
    }
    Ok(z)
}
