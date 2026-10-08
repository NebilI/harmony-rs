use harmorust::{
    embeddings_from_rows, evaluate, run_harmony, simulate_atlas, HarmonyConfig, SimulateSpec,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{Read, Write};

#[derive(Debug, Deserialize)]
struct RunRequest {
    embeddings: Vec<Vec<f64>>,
    batch: Vec<String>,
    #[serde(default)]
    cell_type: Option<Vec<String>>,
    #[serde(default)]
    config: Option<HarmonyConfig>,
}

#[derive(Debug, Serialize)]
struct RunResponse {
    embeddings: Vec<Vec<f64>>,
    n_iter: usize,
    converged: bool,
    n_clusters: usize,
    objective: Vec<f64>,
    before: Option<harmorust::IntegrationMetrics>,
    after: Option<harmorust::IntegrationMetrics>,
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
                batch: atlas.batch,
                cell_type: Some(atlas.cell_type),
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
        _ => {
            eprintln!("harmorust — Rust Harmony (seurust-style kernel port)\n");
            eprintln!("Usage:");
            eprintln!("  harmorust run < request.json");
            eprintln!("  harmorust demo [--cells N] [--batch-strength S] [--seed SEED]");
            eprintln!("  harmorust bench [--cells N]");
        }
    }
    Ok(())
}

fn integrate(req: &RunRequest) -> Result<RunResponse, Box<dyn std::error::Error>> {
    let z = embeddings_from_rows(&req.embeddings)?;
    let cfg = req.config.clone().unwrap_or_default();
    let before = req
        .cell_type
        .as_ref()
        .map(|types| evaluate(z.view(), &req.batch, types));
    let result = run_harmony(z.view(), &req.batch, &cfg)?;
    let after = req.cell_type.as_ref().map(|types| {
        let corr = embeddings_from_rows(&result.embeddings).expect("corrected rows");
        evaluate(corr.view(), &req.batch, types)
    });
    Ok(RunResponse {
        embeddings: result.embeddings,
        n_iter: result.n_iter,
        converged: result.converged,
        n_clusters: result.n_clusters,
        objective: result.objective,
        before,
        after,
    })
}
