export type IntegrationMetrics = {
  batchSeparation: number;
  typeSeparation: number;
  batchLisi: number;
  typeLisi: number;
};

function groupSeparation(z: number[][], labels: string[]): number {
  const n = z.length;
  if (!n) return 0;
  const d = Math.min(4, z[0].length);
  const global = Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) global[j] += z[i][j];
  }
  for (let j = 0; j < d; j++) global[j] /= n;

  const sums = new Map<string, { mean: number[]; count: number }>();
  for (let i = 0; i < n; i++) {
    const entry = sums.get(labels[i]) ?? { mean: Array(d).fill(0), count: 0 };
    for (let j = 0; j < d; j++) entry.mean[j] += z[i][j];
    entry.count += 1;
    sums.set(labels[i], entry);
  }
  let sep = 0;
  for (const { mean, count } of sums.values()) {
    let dist = 0;
    for (let j = 0; j < d; j++) {
      const m = mean[j] / count;
      dist += (m - global[j]) ** 2;
    }
    sep += Math.sqrt(dist);
  }
  return sep / sums.size;
}

function meanLisi(z: number[][], labels: string[], k: number): number {
  const n = z.length;
  if (n < 3) return 1;
  const d = Math.min(2, z[0].length);
  k = Math.max(1, Math.min(k, n - 1));
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const dists: { dist: number; j: number }[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      let dist = 0;
      for (let p = 0; p < d; p++) {
        const diff = z[i][p] - z[j][p];
        dist += diff * diff;
      }
      dists.push({ dist, j });
    }
    dists.sort((a, b) => a.dist - b.dist);
    const counts = new Map<string, number>();
    for (let t = 0; t < k; t++) {
      const label = labels[dists[t].j];
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    let simpson = 0;
    for (const c of counts.values()) {
      const p = c / k;
      simpson += p * p;
    }
    acc += simpson > 0 ? 1 / simpson : 1;
  }
  return acc / n;
}

export function evaluate(
  embeddings: number[][],
  batch: string[],
  cellType: string[],
): IntegrationMetrics {
  return {
    batchSeparation: groupSeparation(embeddings, batch),
    typeSeparation: groupSeparation(embeddings, cellType),
    batchLisi: meanLisi(embeddings, batch, 30),
    typeLisi: meanLisi(embeddings, cellType, 30),
  };
}
