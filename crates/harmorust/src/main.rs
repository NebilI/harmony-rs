use harmorust::compare::{print_report, run_compare, CompareOptions, CompareProtocol};
use harmorust::parity::{
    assert_parity_within, fmt_sci, print_report as print_parity_report, run_parity, ParityOptions,
    ParityProtocol, PARITY_MAX_ABS,
};
use harmorust::{
    embeddings_from_rows, evaluate, run_harmony, run_harmony_multi, simulate_atlas, HarmonyConfig,
    HarmonyResult, SimulateSpec,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{Read, Write};

/// JSON number or array of numbers (jsonlite may unbox length-1 vectors).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NumList {
    Scalar(f64),
    Vec(Vec<f64>),
}

#[derive(Debug, Deserialize)]
struct RunRequest {
    /// Cells × PCs.
    embeddings: Vec<Vec<f64>>,
    /// Single batch vector (legacy / simple API).
    #[serde(default)]
    batch: Option<Vec<String>>,
    /// One or more covariates (Harmony `vars_use`). Overrides `batch` when set.
    #[serde(default)]
    covariates: Option<Vec<Vec<String>>>,
    #[serde(default)]
    cell_type: Option<Vec<String>>,
    /// Diversity penalty: one value or one per covariate.
    #[serde(default)]
    theta: Option<NumList>,
    /// Ridge penalty: omit / null for automatic lambda; one value or one per covariate.
    #[serde(default)]
    lambda: Option<NumList>,
    #[serde(default)]
    config: Option<HarmonyConfig>,
}

#[derive(Debug, Serialize)]
struct RunResponse {
    embeddings: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    o: Vec<Vec<f64>>,
    n_iter: usize,
    converged: bool,
    n_clusters: usize,
    objective: Vec<f64>,
    objective_kmeans: Vec<f64>,
    kmeans_rounds: Vec<usize>,
    before: Option<harmorust::IntegrationMetrics>,
    after: Option<harmorust::IntegrationMetrics>,
}

impl From<HarmonyResult> for RunResponse {
    fn from(result: HarmonyResult) -> Self {
        Self {
            embeddings: result.embeddings,
            r: result.r,
            y: result.y,
            o: result.o,
            n_iter: result.n_iter,
            converged: result.converged,
            n_clusters: result.n_clusters,
            objective: result.objective,
            objective_kmeans: result.objective_kmeans,
            kmeans_rounds: result.kmeans_rounds,
            before: None,
            after: None,
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    match cmd.as_str() {
        "run" => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            let req: RunRequest = serde_json::from_str(&buf)?;
            let out = integrate(&req)?;
            let mut stdout = std::io::stdout();
            serde_json::to_writer(&mut stdout, &out)?;
            stdout.write_all(b"\n")?;
        }
        "demo" => {
            let mut cells = 600usize;
            let mut batch_strength = 2.4f64;
            let mut seed = 7u64;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--cells" => cells = args.next().ok_or("missing --cells")?.parse()?,
                    "--batch-strength" => {
                        batch_strength = args.next().ok_or("missing --batch-strength")?.parse()?
                    }
                    "--seed" => seed = args.next().ok_or("missing --seed")?.parse()?,
                    other => return Err(format!("unknown flag {other}").into()),
                }
            }
            let spec = SimulateSpec {
                n_cells: cells,
                batch_strength,
                seed,
                ..SimulateSpec::default()
            };
            let atlas = simulate_atlas(&spec);
            let req = RunRequest {
                embeddings: atlas.embeddings,
                batch: Some(atlas.batch),
                covariates: None,
                cell_type: Some(atlas.cell_type),
                theta: None,
                lambda: Some(NumList::Scalar(1.0)),
                config: Some(HarmonyConfig {
                    nclust: Some(9),
                    seed,
                    ..HarmonyConfig::default()
                }),
            };
            let out = integrate(&req)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        "bench" => {
            let mut cells = 4000usize;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--cells" => cells = args.next().ok_or("missing --cells")?.parse()?,
                    other => return Err(format!("unknown flag {other}").into()),
                }
            }
            let spec = SimulateSpec {
                n_cells: cells,
                n_pcs: 20,
                n_batches: 4,
                n_types: 5,
                batch_strength: 1.8,
                seed: 1,
            };
            let atlas = simulate_atlas(&spec);
            let z = embeddings_from_rows(&atlas.embeddings)?;
            let cfg = HarmonyConfig {
                nclust: Some(spec.n_types),
                max_iter: 8,
                seed: 1,
                ..HarmonyConfig::default()
            };
            let start = std::time::Instant::now();
            let out = run_harmony(z.view(), &atlas.batch, &cfg)?;
            let elapsed = start.elapsed();
            println!(
                "cells={} pcs={} iters={} converged={} {:.2?}",
                cells, spec.n_pcs, out.n_iter, out.converged, elapsed
            );
        }
        "compare" => {
            let mut protocol = CompareProtocol::default();
            let mut opts = CompareOptions::default();
            let mut json = false;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--cells" => {
                        protocol.cell_counts =
                            parse_usize_list(&args.next().ok_or("missing --cells")?)?
                    }
                    "--pcs" => protocol.n_pcs = args.next().ok_or("missing --pcs")?.parse()?,
                    "--batches" => {
                        protocol.n_batches = args.next().ok_or("missing --batches")?.parse()?
                    }
                    "--types" => protocol.n_types = args.next().ok_or("missing --types")?.parse()?,
                    "--warmup" => protocol.warmup = args.next().ok_or("missing --warmup")?.parse()?,
                    "--repeats" => {
                        protocol.repeats = args.next().ok_or("missing --repeats")?.parse()?
                    }
                    "--max-iter" => {
                        protocol.max_iter = args.next().ok_or("missing --max-iter")?.parse()?
                    }
                    "--ncores" => protocol.ncores = args.next().ok_or("missing --ncores")?.parse()?,
                    "--nclust" => {
                        let raw = args.next().ok_or("missing --nclust")?;
                        protocol.nclust = if raw == "auto" {
                            None
                        } else {
                            Some(raw.parse()?)
                        };
                    }
                    "--seed" => protocol.seed = args.next().ok_or("missing --seed")?.parse()?,
                    "--rscript" => {
                        opts.rscript = args.next().ok_or("missing --rscript")?.into();
                    }
                    "--rust-only" => opts.rust_only = true,
                    "--json" => json = true,
                    "--early-stop" => protocol.early_stop = true,
                    other => return Err(format!("unknown flag {other}").into()),
                }
            }
            std::env::set_var("RAYON_NUM_THREADS", protocol.ncores.max(1).to_string());
            let report = run_compare(&protocol, &opts)?;
            print_report(&report, json);
        }
        "parity" => {
            let mut protocol = ParityProtocol::default();
            let mut opts = ParityOptions::default();
            let mut json = false;
            let mut check = false;
            let mut cells_explicit = false;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--cells" => {
                        cells_explicit = true;
                        protocol.compare.cell_counts =
                            parse_usize_list(&args.next().ok_or("missing --cells")?)?
                    }
                    "--pcs" => {
                        protocol.compare.n_pcs = args.next().ok_or("missing --pcs")?.parse()?
                    }
                    "--batches" => {
                        protocol.compare.n_batches =
                            args.next().ok_or("missing --batches")?.parse()?
                    }
                    "--types" => {
                        protocol.compare.n_types = args.next().ok_or("missing --types")?.parse()?
                    }
                    "--max-iter" => {
                        protocol.compare.max_iter =
                            args.next().ok_or("missing --max-iter")?.parse()?
                    }
                    "--ncores" => {
                        protocol.compare.ncores = args.next().ok_or("missing --ncores")?.parse()?
                    }
                    "--nclust" => {
                        protocol.nclusts =
                            parse_nclust_list(&args.next().ok_or("missing --nclust")?)?
                    }
                    "--seed" => {
                        protocol.compare.seed = args.next().ok_or("missing --seed")?.parse()?
                    }
                    "--rscript" => {
                        opts.rscript = args.next().ok_or("missing --rscript")?.into();
                    }
                    "--json" => json = true,
                    "--check" => check = true,
                    "--early-stop" => protocol.compare.early_stop = true,
                    other => return Err(format!("unknown flag {other}").into()),
                }
            }
            if check && !cells_explicit {
                protocol.compare.cell_counts = ParityProtocol::pr_gate().compare.cell_counts;
            }
            std::env::set_var(
                "RAYON_NUM_THREADS",
                protocol.compare.ncores.max(1).to_string(),
            );
            let report = run_parity(&protocol, &opts)?;
            print_parity_report(&report, json);
            if check {
                assert_parity_within(&report)?;
                if !json {
                    eprintln!(
                        "parity gate passed: gated max |Δ| ≤ {}",
                        fmt_sci(PARITY_MAX_ABS)
                    );
                }
            }
        }
        _ => {
            eprintln!("harmorust — Rust Harmony (seurust-style kernel port)\n");
            eprintln!("Usage:");
            eprintln!("  harmorust run < request.json");
            eprintln!("  harmorust demo [--cells N] [--batch-strength S] [--seed SEED]");
            eprintln!("  harmorust bench [--cells N]");
            eprintln!("  harmorust compare [--cells N,N,...] [--warmup W] [--repeats R] [--ncores N]");
            eprintln!(
                "  harmorust parity [--cells N,N,...] [--nclust K,K,...] [--check] [--json]"
            );
        }
    }
    Ok(())
}

fn parse_nclust_list(raw: &str) -> Result<Vec<Option<usize>>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part == "auto" {
            out.push(None);
        } else {
            out.push(Some(part.parse()?));
        }
    }
    if out.is_empty() {
        return Err("need at least one nclust setting".into());
    }
    Ok(out)
}

fn parse_usize_list(raw: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let cells: Result<Vec<usize>, _> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect();
    let cells = cells?;
    if cells.is_empty() {
        return Err("need at least one cell count".into());
    }
    Ok(cells)
}

fn integrate(req: &RunRequest) -> Result<RunResponse, Box<dyn std::error::Error>> {
    let z = embeddings_from_rows(&req.embeddings)?;
    let mut cfg = req.config.clone().unwrap_or_default();

    let covariates: Vec<Vec<String>> = if let Some(covs) = &req.covariates {
        if covs.is_empty() {
            return Err("covariates must be non-empty when provided".into());
        }
        covs.clone()
    } else if let Some(batch) = &req.batch {
        vec![batch.clone()]
    } else {
        return Err("need batch or covariates".into());
    };

    let lambda_vec = req.lambda.as_ref().map(|l| match l {
        NumList::Scalar(x) => vec![*x],
        NumList::Vec(v) => v.clone(),
    });
    if let Some(lambda) = &lambda_vec {
        if let Some(first) = lambda.first() {
            // Length > 1 is applied below via run_harmony_multi.
            cfg.lambda = Some(*first);
        }
    }

    let theta = req
        .theta
        .as_ref()
        .map(|t| match t {
            NumList::Scalar(x) => vec![*x],
            NumList::Vec(v) => v.clone(),
        })
        .unwrap_or_else(|| vec![cfg.theta; covariates.len()]);

    let result = if covariates.len() == 1 && lambda_vec.as_ref().map(|l| l.len()).unwrap_or(1) <= 1
    {
        run_harmony(z.view(), &covariates[0], &cfg)?
    } else {
        run_harmony_multi(
            z.view(),
            &covariates,
            &theta,
            lambda_vec.as_deref(),
            &cfg,
        )?
    };

    let before = req
        .cell_type
        .as_ref()
        .map(|types| evaluate(z.view(), &covariates[0], types));
    let after = req.cell_type.as_ref().map(|types| {
        let corr = embeddings_from_rows(&result.embeddings).expect("corrected rows");
        evaluate(corr.view(), &covariates[0], types)
    });

    let mut out = RunResponse::from(result);
    out.before = before;
    out.after = after;
    Ok(out)
}
