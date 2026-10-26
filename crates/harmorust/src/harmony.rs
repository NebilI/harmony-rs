use crate::cluster::{init_r, kmeans_cosine, objective, observed_expected_index, update_r};
use crate::config::HarmonyConfig;
use crate::correct::{default_lambda, moe_correct_ridge};
use crate::design::{encode_covariates, encode_labels, expand_lambda, expand_theta, Design};
use crate::error::{HarmonyError, Result};
use crate::linalg::{l2_normalize_columns, zeros_f64_f};
use crate::r_rng::RRng;
use crate::timers::{PhaseTimers, ScopeTimer};
use ndarray::{Array1, Array2, ArrayView2};
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
    /// Expected cluster×level counts E.
    #[serde(default)]
    pub e: Vec<Vec<f64>>,
    /// Advanced R `.Random.seed` after the run, when the caller supplied one.
    #[serde(default)]
    pub rng_state: Option<Vec<i32>>,
    pub n_iter: usize,
    pub converged: bool,
    pub n_clusters: usize,
    /// Outer-round objectives (Harmony's `objective_harmony`).
    pub objective: Vec<f64>,
    /// Inner clustering objectives (Harmony's `objective_kmeans`), used by
    /// `HarmonyConvergencePlot`.
    pub objective_kmeans: Vec<f64>,
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
    /// R Mersenne-Twister shared by k-means++ and block shuffles. Harmony's
    /// Module methods advance the C-level RNG from `set.seed` without writing
    /// `.Random.seed` back, so the stream continues across init and every
    /// `update_R` for the whole `RunHarmony` call.
    rng: RRng,
    pub timers: PhaseTimers,
    time_phases: bool,
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
        // Same integer seed as R `set.seed`, or a full `.Random.seed` vector
        // when the R package forwarded the session RNG.
        let mut rng = if let Some(state) = config.rng_state.as_deref() {
            RRng::from_r_seed(state).map_err(HarmonyError::msg)?
        } else {
            RRng::from_harmony_seed(config.seed)
        };
        let time_phases = PhaseTimers::enabled();
        let mut timers = PhaseTimers::default();

        // Internal layout is PCs × cells, matching the C++ Harmony object.
        // Store Z / R / dist column-major so each cell is contiguous.
        let z_orig = into_column_major(embeddings.t().mapv(|v| f64::from(v as f32)));
        let z_corr = into_column_major(
            l2_normalize_columns(z_orig.view()).mapv(|v| f64::from(v as f32)),
        );
        let kmeans_t = ScopeTimer::start();
        let y = into_column_major(kmeans_cosine(z_corr.view(), k, &mut rng, 10));
        if time_phases {
            timers.add("kmeans", kmeans_t.elapsed_ms());
        }
        let sigma = Array1::from_elem(k, config.sigma);
        let softmax_t = ScopeTimer::start();
        let (r, dist) = init_r(y.view(), z_corr.view(), &sigma);
        if time_phases {
            timers.add("softmax", softmax_t.elapsed_ms());
        }
        let r = into_column_major(r);
        let dist = into_column_major(dist);
        let (o, e) = observed_expected_index(r.view(), &design);

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
            rng,
            timers,
            time_phases,
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
            &self.design,
            &self.theta,
            &self.sigma,
        )
    }

    /// One clustering round (re-init R after a correction, then online `update_R`).
    pub fn cluster_round(&mut self) {
        // After a correction, re-estimate R from the new Z_corr and current Y.
        if !self.kmeans_rounds.is_empty() {
            self.z_corr = l2_normalize_columns(self.z_corr.view());
            let softmax_t = ScopeTimer::start();
            let (r, dist) = init_r(self.y.view(), self.z_corr.view(), &self.sigma);
            if self.time_phases {
                self.timers.add("softmax", softmax_t.elapsed_ms());
            }
            self.r = into_column_major(r);
            self.dist = into_column_major(dist);
            let (o, e) = observed_expected_index(self.r.view(), &self.design);
            self.o = o;
            self.e = e;
        }

        // `init_r` already stored the distance-only softmax in `self.r`.
        // Reuse it across inner `update_r` passes instead of calling `exp`
        // again.
        let r_base = self.r.clone();
        let mut used = 0usize;
        for iter in 0..self.config.max_iter_cluster {
            used = iter + 1;
            let upd_t = ScopeTimer::start();
            update_r(
                &mut self.r,
                r_base.view(),
                &self.design,
                &self.theta,
                &mut self.o,
                &mut self.e,
                self.config.block_size,
                &mut self.rng,
            );
            if self.time_phases {
                self.timers.add("update_r", upd_t.elapsed_ms());
            }
            let obj_t = ScopeTimer::start();
            let obj = self.current_objective();
            if self.time_phases {
                self.timers.add("objective", obj_t.elapsed_ms());
            }
            self.objective_kmeans.push(obj);
            // Harmony: `if (iter > window_size)` with window_size = 3.
            if iter > 3 && self.cluster_converged() {
                break;
            }
        }
        self.kmeans_rounds.push(used);
        self.objective_harmony
            .push(*self.objective_kmeans.last().unwrap());
    }

    /// One mixture-of-experts ridge correction from `Z_orig`.
    pub fn correct_round(&mut self) -> Result<()> {
        let ridge_t = ScopeTimer::start();
        self.z_corr = into_column_major(moe_correct_ridge(
            self.z_orig.view(),
            self.r.view(),
            self.o.view(),
            self.e.view(),
            &self.design,
            self.lambda.as_deref(),
            self.config.alpha,
            self.config.batch_prop_cutoff,
            &mut self.y,
        )?);
        if self.time_phases {
            self.timers.add("ridge", ridge_t.elapsed_ms());
        }
        Ok(())
    }

    /// Replace Y with an external (Harmony) centroid matrix and re-init R.
    /// Used for teacher-forced per-stage diffs.
    pub fn teacher_force_y(&mut self, y: Array2<f64>) {
        self.y = into_column_major(y);
        let (r, dist) = init_r(self.y.view(), self.z_corr.view(), &self.sigma);
        self.r = into_column_major(r);
        self.dist = into_column_major(dist);
        let (o, e) = observed_expected_index(self.r.view(), &self.design);
        self.o = o;
        self.e = e;
    }

    /// Snapshot the current state as a [`HarmonyResult`].
    pub fn snapshot(&self, n_iter: usize, converged: bool) -> HarmonyResult {
        HarmonyResult {
            embeddings: array_to_cells_by_pc(&self.z_corr),
            r: array_to_rows(&self.r),
            y: array_to_rows(&self.y),
            o: array_to_rows(&self.o),
            e: array_to_rows(&self.e),
            rng_state: Some(self.rng.to_r_seed()),
            n_iter,
            converged,
            n_clusters: self.y.ncols(),
            objective: self.objective_harmony.clone(),
            objective_kmeans: self.objective_kmeans.clone(),
            kmeans_rounds: self.kmeans_rounds.clone(),
        }
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
        let mut n_iter = 0usize;
        let mut converged = false;
        for _ in 0..self.config.max_iter {
            n_iter += 1;
            self.cluster_round();
            self.correct_round()?;
            if self.harmony_converged() {
                converged = true;
                break;
            }
        }

        if self.time_phases {
            eprintln!(
                "harmorust timers: kmeans={:.2} softmax={:.2} update_r={:.2} objective={:.2} ridge={:.2} total={:.2} ms",
                self.timers.kmeans_ms,
                self.timers.softmax_ms,
                self.timers.update_r_ms,
                self.timers.objective_ms,
                self.timers.ridge_ms,
                self.timers.total_ms()
            );
        }

        let mut out = self.snapshot(n_iter, converged);
        if self.config.rng_state.is_none() {
            out.rng_state = None;
        }
        Ok(out)
    }
}

fn into_column_major(a: Array2<f64>) -> Array2<f64> {
    if a.strides().first().copied() == Some(1) {
        return a;
    }
    let mut out = zeros_f64_f(a.nrows(), a.ncols());
    out.assign(&a);
    out
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
