export type HarmonyConfig = {
  theta: number;
  sigma: number;
  lambda: number;
  nclust: number;
  maxIter: number;
  maxIterCluster: number;
  seed: number;
};

export type HarmonyResult = {
  embeddings: number[][];
  nIter: number;
  nClusters: number;
  objective: number[];
};

function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function transpose(z: number[][]): number[][] {
  const n = z.length;
  const d = z[0].length;
  const out = Array.from({ length: d }, () => Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) out[j][i] = z[i][j];
  }
  return out;
}

function l2NormalizeCols(z: number[][]): number[][] {
  const d = z.length;
  const n = z[0].length;
  const out = z.map((row) => row.slice());
  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = 0; i < d; i++) norm += out[i][j] * out[i][j];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < d; i++) out[i][j] /= norm;
    }
  }
  return out;
}

function invert(a: number[][]): number[][] {
  const n = a.length;
  const m = a.map((row, i) => {
    const r = row.slice();
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(m[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(m[r][col]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best < 1e-18) throw new Error("singular ridge covariance");
    if (pivot !== col) {
      const tmp = m[col];
      m[col] = m[pivot];
      m[pivot] = tmp;
    }
    const diag = m[col][col];
    for (let j = col; j < 2 * n; j++) m[col][j] /= diag;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col];
      if (f === 0) continue;
      for (let j = col; j < 2 * n; j++) m[r][j] -= f * m[col][j];
    }
  }
  return m.map((row) => row.slice(n));
}

function cosineDist(y: number[][], z: number[][]): number[][] {
  const k = y[0].length;
  const n = z[0].length;
  const d = y.length;
  const dist = Array.from({ length: k }, () => Array<number>(n).fill(0));
  for (let c = 0; c < k; c++) {
    for (let j = 0; j < n; j++) {
      let sim = 0;
      for (let i = 0; i < d; i++) sim += y[i][c] * z[i][j];
      dist[c][j] = 2 * (1 - sim);
    }
  }
  return dist;
}

function softmaxNegDist(dist: number[][], sigma: number[]): number[][] {
  const k = dist.length;
  const n = dist[0].length;
  const r = Array.from({ length: k }, () => Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let maxLogit = -Infinity;
    for (let i = 0; i < k; i++) maxLogit = Math.max(maxLogit, -dist[i][j] / sigma[i]);
    let sum = 0;
    for (let i = 0; i < k; i++) {
      const v = Math.exp(-dist[i][j] / sigma[i] - maxLogit);
      r[i][j] = v;
      sum += v;
    }
    for (let i = 0; i < k; i++) r[i][j] /= sum || 1;
  }
  return r;
}

function kmeansCosine(z: number[][], k: number, rng: () => number): number[][] {
  const d = z.length;
  const n = z[0].length;
  k = Math.max(1, Math.min(k, n));
  const y = Array.from({ length: d }, () => Array<number>(k).fill(0));
  const chosen: number[] = [];
  const first = Math.floor(rng() * n);
  chosen.push(first);
  for (let i = 0; i < d; i++) y[i][0] = z[i][first];

  for (let c = 1; c < k; c++) {
    const minDist = Array(n).fill(Infinity);
    for (let j = 0; j < n; j++) {
      for (let prev = 0; prev < c; prev++) {
        let sim = 0;
        for (let i = 0; i < d; i++) sim += y[i][prev] * z[i][j];
        minDist[j] = Math.min(minDist[j], Math.abs(2 * (1 - sim)) || 1e-12);
      }
    }
    let bestIdx = 0;
    let bestScore = Infinity;
    for (let j = 0; j < n; j++) {
      if (chosen.includes(j)) continue;
      const u = Math.min(1, Math.max(1e-12, rng()));
      const score = -Math.log(u) / minDist[j];
      if (score < bestScore) {
        bestScore = score;
        bestIdx = j;
      }
    }
    chosen.push(bestIdx);
    for (let i = 0; i < d; i++) y[i][c] = z[i][bestIdx];
  }

  for (let iter = 0; iter < 8; iter++) {
    const dist = cosineDist(y, z);
    const next = Array.from({ length: d }, () => Array<number>(k).fill(0));
    const counts = Array(k).fill(0);
    for (let j = 0; j < n; j++) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        if (dist[c][j] < bestD) {
          bestD = dist[c][j];
          best = c;
        }
      }
      counts[best] += 1;
      for (let i = 0; i < d; i++) next[i][best] += z[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let i = 0; i < d; i++) next[i][c] /= counts[c];
      } else {
        for (let i = 0; i < d; i++) next[i][c] = y[i][c];
      }
    }
    const normed = l2NormalizeCols(next);
    for (let i = 0; i < d; i++) y[i] = normed[i];
  }
  return l2NormalizeCols(y);
}

function observedExpected(r: number[][], phi: number[][], prB: number[]) {
  const k = r.length;
  const n = r[0].length;
  const b = phi.length;
  const o = Array.from({ length: k }, () => Array<number>(b).fill(0));
  const e = Array.from({ length: k }, () => Array<number>(b).fill(0));
  for (let i = 0; i < k; i++) {
    let rk = 0;
    for (let j = 0; j < n; j++) {
      rk += r[i][j];
      for (let bb = 0; bb < b; bb++) o[i][bb] += r[i][j] * phi[bb][j];
    }
    for (let bb = 0; bb < b; bb++) e[i][bb] = rk * prB[bb];
  }
  return { o, e };
}

function updateR(
  r: number[][],
  dist: number[][],
  phi: number[][],
  prB: number[],
  theta: number[],
  sigma: number[],
  o: number[][],
  e: number[][],
  blockSize: number,
  rng: () => number,
) {
  const n = r[0].length;
  const k = r.length;
  const b = phi.length;
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const nBlocks = Math.max(1, Math.ceil(1 / blockSize));
  const per = Math.max(1, Math.floor(n * blockSize));

  for (let block = 0; block < nBlocks; block++) {
    const start = block * per;
    if (start >= n) break;
    const end = block === nBlocks - 1 ? n : Math.min(n, (block + 1) * per);
    const cells = order.slice(start, end);

    for (const j of cells) {
      for (let i = 0; i < k; i++) {
        for (let bb = 0; bb < b; bb++) {
          e[i][bb] -= r[i][j] * prB[bb];
          o[i][bb] -= r[i][j] * phi[bb][j];
        }
      }
    }

    for (const j of cells) {
      let maxLogit = -Infinity;
      const logits = Array(k);
      for (let i = 0; i < k; i++) {
        logits[i] = -dist[i][j] / sigma[i];
        maxLogit = Math.max(maxLogit, logits[i]);
      }
      let sum = 0;
      const col = Array(k);
      for (let i = 0; i < k; i++) {
        col[i] = Math.exp(logits[i] - maxLogit);
        sum += col[i];
      }
      for (let i = 0; i < k; i++) col[i] /= sum || 1;
      for (let i = 0; i < k; i++) {
        let penalty = 1;
        for (let bb = 0; bb < b; bb++) {
          if (phi[bb][j] === 0) continue;
          penalty *= Math.pow((2 * e[i][bb] + 1) / (o[i][bb] + e[i][bb] + 1), theta[bb]);
        }
        col[i] *= penalty;
      }
      sum = col.reduce((a, v) => a + v, 0);
      for (let i = 0; i < k; i++) {
        r[i][j] = col[i] / (sum || 1);
        for (let bb = 0; bb < b; bb++) {
          e[i][bb] = Math.max(0, e[i][bb] + r[i][j] * prB[bb]);
          o[i][bb] = Math.max(0, o[i][bb] + r[i][j] * phi[bb][j]);
        }
      }
    }
  }
}

export function runHarmony(
  embeddings: number[][],
  batch: string[],
  config: HarmonyConfig,
): HarmonyResult {
  const n = embeddings.length;
  if (n < 6) throw new Error("Need at least 6 cells");
  const d = embeddings[0].length;
  const levels = Array.from(new Set(batch));
  if (levels.length < 2) throw new Error("Need at least 2 batches");
  const code = batch.map((label) => levels.indexOf(label));
  const b = levels.length;
  const phi = Array.from({ length: b }, () => Array<number>(n).fill(0));
  const batchSizes = Array(b).fill(0);
  for (let j = 0; j < n; j++) {
    phi[code[j]][j] = 1;
    batchSizes[code[j]] += 1;
  }
  const prB = batchSizes.map((s) => s / n);
  const k = Math.max(1, Math.min(config.nclust, n));
  const rng = mulberry32(config.seed);

  let zOrig = transpose(embeddings);
  let zCorr = l2NormalizeCols(zOrig);
  let y = kmeansCosine(zCorr, k, rng);
  const sigma = Array(k).fill(config.sigma);
  const theta = Array(b).fill(config.theta);
  let dist = cosineDist(y, zCorr);
  let r = softmaxNegDist(dist, sigma);
  let { o, e } = observedExpected(r, phi, prB);
  const objective: number[] = [];
  const blockSize = n < 40 ? 0.2 : 0.05;

  for (let iter = 0; iter < config.maxIter; iter++) {
    if (iter > 0) {
      zCorr = l2NormalizeCols(zCorr);
      dist = cosineDist(y, zCorr);
      r = softmaxNegDist(dist, sigma);
      ({ o, e } = observedExpected(r, phi, prB));
    }
    for (let inner = 0; inner < config.maxIterCluster; inner++) {
      updateR(r, dist, phi, prB, theta, sigma, o, e, blockSize, rng);
    }

    zCorr = zOrig.map((row) => row.slice());
    for (let cluster = 0; cluster < k; cluster++) {
      const keep: number[] = [];
      for (let bb = 0; bb < b; bb++) {
        const avg = batchSizes[bb] > 0 ? o[cluster][bb] / batchSizes[bb] : 0;
        if (avg > 1e-5) keep.push(bb);
      }
      if (keep.length < 2) continue;
      const m = keep.length + 1;
      const phiCov = Array.from({ length: m }, () => Array<number>(m).fill(0));
      let rSum = 0;
      for (let j = 0; j < n; j++) rSum += r[cluster][j];
      phiCov[0][0] = rSum;
      for (let t = 0; t < keep.length; t++) {
        const ob = o[cluster][keep[t]];
        phiCov[0][t + 1] = ob;
        phiCov[t + 1][0] = ob;
        phiCov[t + 1][t + 1] = ob + config.lambda;
      }
      const inv = invert(phiCov);
      const rhs = Array.from({ length: m }, () => Array<number>(d).fill(0));
      for (let j = 0; j < n; j++) {
        const w = r[cluster][j];
        if (!w) continue;
        for (let i = 0; i < d; i++) rhs[0][i] += w * zOrig[i][j];
        const t = keep.indexOf(code[j]);
        if (t >= 0) {
          for (let i = 0; i < d; i++) rhs[t + 1][i] += w * zOrig[i][j];
        }
      }
      const wCoef = Array.from({ length: m }, () => Array<number>(d).fill(0));
      for (let p = 0; p < m; p++) {
        for (let i = 0; i < d; i++) {
          let s = 0;
          for (let q = 0; q < m; q++) s += inv[p][q] * rhs[q][i];
          wCoef[p][i] = s;
        }
      }
      for (let i = 0; i < d; i++) y[i][cluster] = wCoef[0][i];
      for (let j = 0; j < n; j++) {
        const w = r[cluster][j];
        if (!w) continue;
        const t = keep.indexOf(code[j]);
        if (t < 0) continue;
        for (let i = 0; i < d; i++) zCorr[i][j] -= wCoef[t + 1][i] * w;
      }
    }
    y = l2NormalizeCols(y);
    objective.push(iter);
  }

  return {
    embeddings: transpose(zCorr),
    nIter: config.maxIter,
    nClusters: k,
    objective,
  };
}
