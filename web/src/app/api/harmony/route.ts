import { NextResponse } from "next/server";
import { computeDemo, DEFAULT_PARAMS } from "@/lib/run-demo";

export const runtime = "nodejs";

type Body = Partial<{
  nCells: number;
  nTypes: number;
  nBatches: number;
  batchStrength: number;
  theta: number;
  sigma: number;
  lambda: number;
  nclust: number;
  seed: number;
}>;

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Body;
    const result = computeDemo({
      nCells: clamp(body.nCells ?? DEFAULT_PARAMS.nCells, 60, 900),
      nTypes: clamp(body.nTypes ?? DEFAULT_PARAMS.nTypes, 2, 5),
      nBatches: clamp(body.nBatches ?? DEFAULT_PARAMS.nBatches, 2, 4),
      batchStrength: clamp(body.batchStrength ?? DEFAULT_PARAMS.batchStrength, 0, 5),
      theta: clamp(body.theta ?? DEFAULT_PARAMS.theta, 0, 8),
      sigma: clamp(body.sigma ?? DEFAULT_PARAMS.sigma, 0.02, 1),
      lambda: clamp(body.lambda ?? DEFAULT_PARAMS.lambda, 0.05, 10),
      nclust: clamp(body.nclust ?? DEFAULT_PARAMS.nclust, 1, 20),
      seed: body.seed ?? DEFAULT_PARAMS.seed,
    });
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Harmony failed";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
