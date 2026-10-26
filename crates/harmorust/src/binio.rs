//! Length-prefixed binary I/O for the `run-bin` CLI.
//!
//! JSON embeddings dominate drop-in R time at 100k cells. The binary path
//! keeps a small JSON header (config, labels) and writes matrices as raw
//! little-endian `f32` / `f64`.

use crate::error::{HarmonyError, Result};
use crate::harmony::HarmonyResult;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub const MAGIC: &[u8; 4] = b"HRu1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinHeader {
    pub n_cells: usize,
    pub n_pcs: usize,
    /// Covariates as character columns (one vector per `vars_use`).
    pub covariates: Vec<Vec<String>>,
    #[serde(default)]
    pub theta: Option<Vec<f64>>,
    #[serde(default)]
    pub lambda: Option<Vec<f64>>,
    #[serde(default)]
    pub config: Option<crate::config::HarmonyConfig>,
    #[serde(default)]
    pub cell_type: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinMeta {
    pub n_iter: usize,
    pub converged: bool,
    pub n_clusters: usize,
    pub objective: Vec<f64>,
    pub objective_kmeans: Vec<f64>,
    pub kmeans_rounds: Vec<usize>,
    #[serde(default)]
    pub rng_state: Option<Vec<i32>>,
}

pub fn read_request<R: Read>(mut r: R) -> Result<(BinHeader, Vec<Vec<f64>>)> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|e| HarmonyError::msg(format!("run-bin magic: {e}")))?;
    if &magic != MAGIC {
        return Err(HarmonyError::msg(format!(
            "run-bin: bad magic {:x?}, expected HRu1",
            magic
        )));
    }
    let header_len = read_u32(&mut r)? as usize;
    let mut header_buf = vec![0u8; header_len];
    r.read_exact(&mut header_buf)
        .map_err(|e| HarmonyError::msg(format!("run-bin header: {e}")))?;
    let header: BinHeader = serde_json::from_slice(&header_buf)
        .map_err(|e| HarmonyError::msg(format!("run-bin header JSON: {e}")))?;
    let n = header.n_cells;
    let d = header.n_pcs;
    let mut raw = vec![0u8; n.saturating_mul(d).saturating_mul(4)];
    r.read_exact(&mut raw)
        .map_err(|e| HarmonyError::msg(format!("run-bin embeddings: {e}")))?;
    let mut embeddings = vec![vec![0.0f64; d]; n];
    for i in 0..n {
        for j in 0..d {
            let off = (i * d + j) * 4;
            let bits = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
            embeddings[i][j] = f64::from(f32::from_bits(bits));
        }
    }
    Ok((header, embeddings))
}

pub fn write_response<W: Write>(mut w: W, result: &HarmonyResult) -> Result<()> {
    w.write_all(MAGIC)
        .map_err(|e| HarmonyError::msg(e.to_string()))?;
    let meta = BinMeta {
        n_iter: result.n_iter,
        converged: result.converged,
        n_clusters: result.n_clusters,
        objective: result.objective.clone(),
        objective_kmeans: result.objective_kmeans.clone(),
        kmeans_rounds: result.kmeans_rounds.clone(),
        rng_state: result.rng_state.clone(),
    };
    let meta_bytes =
        serde_json::to_vec(&meta).map_err(|e| HarmonyError::msg(e.to_string()))?;
    write_u32(&mut w, meta_bytes.len() as u32)?;
    w.write_all(&meta_bytes)
        .map_err(|e| HarmonyError::msg(e.to_string()))?;
    write_matrix_f32(&mut w, &result.embeddings)?;
    write_matrix_f32(&mut w, &result.r)?;
    write_matrix_f32(&mut w, &result.y)?;
    write_matrix_f32(&mut w, &result.o)?;
    write_matrix_f32(&mut w, &result.e)?;
    Ok(())
}

fn write_matrix_f32<W: Write>(w: &mut W, m: &[Vec<f64>]) -> Result<()> {
    let rows = m.len() as u32;
    let cols = m.first().map(|r| r.len()).unwrap_or(0) as u32;
    write_u32(w, rows)?;
    write_u32(w, cols)?;
    for row in m {
        for &v in row {
            w.write_all(&(v as f32).to_bits().to_le_bytes())
                .map_err(|e| HarmonyError::msg(e.to_string()))?;
        }
    }
    Ok(())
}

pub fn read_matrix_f32<R: Read>(r: &mut R) -> Result<Vec<Vec<f64>>> {
    let rows = read_u32(r)? as usize;
    let cols = read_u32(r)? as usize;
    let mut out = vec![vec![0.0f64; cols]; rows];
    let mut buf = [0u8; 4];
    for i in 0..rows {
        for j in 0..cols {
            r.read_exact(&mut buf)
                .map_err(|e| HarmonyError::msg(e.to_string()))?;
            out[i][j] = f64::from(f32::from_bits(u32::from_le_bytes(buf)));
        }
    }
    Ok(out)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| HarmonyError::msg(e.to_string()))?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())
        .map_err(|e| HarmonyError::msg(e.to_string()))
}

/// Build a request file the R package can write.
pub fn write_request<W: Write>(
    mut w: W,
    header: &BinHeader,
    embeddings: &[Vec<f64>],
) -> Result<()> {
    w.write_all(MAGIC)
        .map_err(|e| HarmonyError::msg(e.to_string()))?;
    let header_bytes =
        serde_json::to_vec(header).map_err(|e| HarmonyError::msg(e.to_string()))?;
    write_u32(&mut w, header_bytes.len() as u32)?;
    w.write_all(&header_bytes)
        .map_err(|e| HarmonyError::msg(e.to_string()))?;
    for row in embeddings {
        for &v in row {
            w.write_all(&(v as f32).to_bits().to_le_bytes())
                .map_err(|e| HarmonyError::msg(e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::HarmonyResult;

    #[test]
    fn request_response_round_trip() {
        let header = BinHeader {
            n_cells: 2,
            n_pcs: 3,
            covariates: vec![vec!["A".into(), "B".into()]],
            theta: Some(vec![2.0]),
            lambda: Some(vec![1.0]),
            config: None,
            cell_type: None,
        };
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let mut buf = Vec::new();
        write_request(&mut buf, &header, &embeddings).unwrap();
        let (got_h, got_e) = read_request(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(got_h.n_cells, 2);
        assert_eq!(got_h.n_pcs, 3);
        assert_eq!(got_e.len(), 2);

        let result = HarmonyResult {
            embeddings: got_e,
            r: vec![vec![1.0, 0.0]],
            y: vec![vec![1.0], vec![0.0], vec![0.0]],
            o: vec![vec![1.0, 1.0]],
            e: vec![vec![1.0, 1.0]],
            rng_state: None,
            n_iter: 1,
            converged: true,
            n_clusters: 1,
            objective: vec![1.0],
            objective_kmeans: vec![1.0],
            kmeans_rounds: vec![2],
        };
        let mut out = Vec::new();
        write_response(&mut out, &result).unwrap();
        assert_eq!(&out[..4], MAGIC);
    }
}
