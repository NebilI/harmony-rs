const TYPE_NAMES = ["T cell", "B cell", "Monocyte", "NK cell", "DC"];
const BATCH_NAMES = ["donor_A", "donor_B", "donor_C", "donor_D"];

export type SimulatedAtlas = {
  embeddings: number[][];
  batch: string[];
  cellType: string[];
  nCells: number;
  nPcs: number;
};

export function simulateAtlas(spec: {
  nCells: number;
  nPcs: number;
  nBatches: number;
  nTypes: number;
  batchStrength: number;
  seed: number;
}): SimulatedAtlas {
  const n = Math.max(12, spec.nCells);
  const d = Math.max(2, spec.nPcs);
  const nTypes = Math.max(2, spec.nTypes);
  const nBatches = Math.max(2, spec.nBatches);
  let t = spec.seed >>> 0;
  const rng = () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };

  const centers = Array.from({ length: nTypes }, (_, ti) => {
    const angle = (ti * 2 * Math.PI) / nTypes;
    const row = Array(d).fill(0);
    row[0] = 3.2 * Math.cos(angle);
    row[1] = 3.2 * Math.sin(angle);
    for (let j = 2; j < d; j++) row[j] = rng() * 0.8 - 0.4;
    return row;
  });

  const embeddings: number[][] = [];
  const batch: string[] = [];
  const cellType: string[] = [];
  for (let i = 0; i < n; i++) {
    const ti = i % nTypes;
    const bi = Math.floor(i / nTypes) % nBatches;
    const angle = (bi * 2 * Math.PI) / nBatches;
    const row = Array(d);
    row[0] = centers[ti][0] + spec.batchStrength * Math.cos(angle) + rng() * 0.7 - 0.35;
    row[1] = centers[ti][1] + spec.batchStrength * Math.sin(angle) + rng() * 0.7 - 0.35;
    for (let j = 2; j < d; j++) row[j] = centers[ti][j] + rng() * 0.5 - 0.25;
    embeddings.push(row);
    batch.push(BATCH_NAMES[bi] ?? `donor_${bi}`);
    cellType.push(TYPE_NAMES[ti] ?? `type_${ti}`);
  }

  return { embeddings, batch, cellType, nCells: n, nPcs: d };
}
