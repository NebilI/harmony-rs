import { runHarmony } from "@/lib/harmony";
import { evaluate, type IntegrationMetrics } from "@/lib/metrics";
import { simulateAtlas } from "@/lib/simulate";

export type DemoParams = {
  nCells: number;
  nTypes: number;
  nBatches: number;
  batchStrength: number;
  theta: number;
  sigma: number;
  lambda: number;
  nclust: number;
  seed: number;
};

export type DemoResult = {
  batch: string[];
  cellType: string[];
  beforeEmbeddings: number[][];
  afterEmbeddings: number[][];
  nIter: number;
  nClusters: number;
  before: IntegrationMetrics;
  after: IntegrationMetrics;
};

export const DEFAULT_PARAMS: DemoParams = {
  nCells: 360,
  nTypes: 3,
  nBatches: 2,
  batchStrength: 2.4,
  theta: 2,
  sigma: 0.2,
  lambda: 1,
  nclust: 3,
  seed: 7,
};

export function computeDemo(params: DemoParams): DemoResult {
  const atlas = simulateAtlas({
    nCells: params.nCells,
    nPcs: 8,
    nBatches: params.nBatches,
    nTypes: params.nTypes,
    batchStrength: params.batchStrength,
    seed: params.seed,
  });
  const before = evaluate(atlas.embeddings, atlas.batch, atlas.cellType);
  const result = runHarmony(atlas.embeddings, atlas.batch, {
    theta: params.theta,
    sigma: params.sigma,
    lambda: params.lambda,
    nclust: params.nclust,
    maxIter: 8,
    maxIterCluster: 6,
    seed: params.seed,
  });
  const after = evaluate(result.embeddings, atlas.batch, atlas.cellType);
  return {
    batch: atlas.batch,
    cellType: atlas.cellType,
    beforeEmbeddings: atlas.embeddings.map((row) => [row[0], row[1]]),
    afterEmbeddings: result.embeddings.map((row) => [row[0], row[1]]),
    nIter: result.nIter,
    nClusters: result.nClusters,
    before,
    after,
  };
}
