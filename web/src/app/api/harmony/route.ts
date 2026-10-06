import { NextResponse } from "next/server";
import { runHarmony } from "@/lib/harmony";
import { evaluate } from "@/lib/metrics";
import { simulateAtlas } from "@/lib/simulate";

export const runtime = "nodejs";

type Body = {
  nCells?: number;
  nPcs?: number;
  nBatches?: number;
  nTypes?: number;
  batchStrength?: number;
  seed?: number;
  theta?: number;
  sigma?: number;
  lambda?: number;
  nclust?: number;
  maxIter?: number;
};

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Body;
    const nCells = clamp(body.nCells ?? 360, 60, 900);
    const nTypes = clamp(body.nTypes ?? 3, 2, 5);
    const nBatches = clamp(body.nBatches ?? 2, 2, 4);
    const atlas = simulateAtlas({
      nCells,
      nPcs: clamp(body.nPcs ?? 8, 2, 16),
      nBatches,
      nTypes,
      batchStrength: clamp(body.batchStrength ?? 2.4, 0, 5),
      seed: body.seed ?? 7,
    });
    const nclust = clamp(body.nclust ?? nTypes, 1, 20);
    const before = evaluate(atlas.embeddings, atlas.batch, atlas.cellType);
    const result = runHarmony(atlas.embeddings, atlas.batch, {
      theta: clamp(body.theta ?? 2, 0, 8),
      sigma: clamp(body.sigma ?? 0.2, 0.02, 1),
      lambda: clamp(body.lambda ?? 1, 0.05, 10),
      nclust,
      maxIter: clamp(body.maxIter ?? 8, 1, 15),
      maxIterCluster: 6,
      seed: body.seed ?? 7,
    });
    const after = evaluate(result.embeddings, atlas.batch, atlas.cellType);
    return NextResponse.json({
      batch: atlas.batch,
      cellType: atlas.cellType,
      beforeEmbeddings: atlas.embeddings.map((row) => [row[0], row[1]]),
      afterEmbeddings: result.embeddings.map((row) => [row[0], row[1]]),
      nIter: result.nIter,
      nClusters: result.nClusters,
      before,
      after,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Harmony failed";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
