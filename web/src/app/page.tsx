"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { MetricCard } from "@/components/metric-card";
import { ScatterPlot } from "@/components/scatter-plot";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import type { IntegrationMetrics } from "@/lib/metrics";

type Result = {
  batch: string[];
  cellType: string[];
  beforeEmbeddings: number[][];
  afterEmbeddings: number[][];
  nIter: number;
  nClusters: number;
  before: IntegrationMetrics;
  after: IntegrationMetrics;
};

const DEFAULTS = {
  nCells: 360,
  nTypes: 3,
  nBatches: 2,
  batchStrength: 2.4,
  theta: 2,
  sigma: 0.2,
  lambda: 1,
  nclust: 3,
};

export default function Home() {
  const [nCells, setNCells] = useState(DEFAULTS.nCells);
  const [nTypes, setNTypes] = useState(DEFAULTS.nTypes);
  const [nBatches, setNBatches] = useState(DEFAULTS.nBatches);
  const [batchStrength, setBatchStrength] = useState(DEFAULTS.batchStrength);
  const [theta, setTheta] = useState(DEFAULTS.theta);
  const [sigma, setSigma] = useState(DEFAULTS.sigma);
  const [lambda, setLambda] = useState(DEFAULTS.lambda);
  const [nclust, setNclust] = useState(DEFAULTS.nclust);
  const [colorBy, setColorBy] = useState("batch");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/harmony", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nCells,
          nTypes,
          nBatches,
          batchStrength,
          theta,
          sigma,
          lambda,
          nclust,
          seed: 7,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "Harmony failed");
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Harmony failed");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [nCells, nTypes, nBatches, batchStrength, theta, sigma, lambda, nclust]);

  useEffect(() => {
    void run();
    // First paint only — later runs are explicit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const labels = result
    ? colorBy === "batch"
      ? result.batch
      : result.cellType
    : [];

  return (
    <div className="min-h-full bg-background">
      <header className="border-b">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 sm:px-6">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">harmorust</Badge>
            <Badge variant="outline">companion to seurust</Badge>
          </div>
          <h1 className="font-heading max-w-3xl text-3xl font-medium tracking-tight sm:text-4xl">
            Harmony in Rust, the same way seurust ports Seurat
          </h1>
          <p className="max-w-3xl text-pretty text-muted-foreground">
            Harmony only needs the PCA embedding Seurat already computed. This
            demo builds a two-donor atlas with a planted batch shift, then runs
            the Harmony loop — soft k-means, diversity penalty, mixture-of-experts
            ridge — that lives in{" "}
            <code className="font-mono text-xs">crates/harmorust</code>. If the
            batches collapse and the cell types stay put, the port works.
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 px-4 py-8 sm:px-6 lg:grid-cols-[minmax(0,18rem)_1fr]">
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Synthetic atlas</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <SliderField
              label="Cells"
              value={nCells}
              min={80}
              max={720}
              step={20}
              format={(v) => String(v)}
              onChange={setNCells}
            />
            <SliderField
              label="Cell types"
              value={nTypes}
              min={2}
              max={5}
              step={1}
              format={(v) => String(v)}
              onChange={(v) => {
                setNTypes(v);
                setNclust(v);
              }}
            />
            <SliderField
              label="Donors"
              value={nBatches}
              min={2}
              max={4}
              step={1}
              format={(v) => String(v)}
              onChange={setNBatches}
            />
            <SliderField
              label="Batch shift"
              value={batchStrength}
              min={0}
              max={4}
              step={0.1}
              format={(v) => v.toFixed(1)}
              onChange={setBatchStrength}
            />
            <Separator />
            <SliderField
              label="Clusters (nclust)"
              value={nclust}
              min={1}
              max={12}
              step={1}
              format={(v) => String(v)}
              onChange={setNclust}
            />
            <SliderField
              label="Theta"
              value={theta}
              min={0}
              max={6}
              step={0.5}
              format={(v) => v.toFixed(1)}
              onChange={setTheta}
            />
            <SliderField
              label="Sigma"
              value={sigma}
              min={0.05}
              max={0.6}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={setSigma}
            />
            <SliderField
              label="Lambda"
              value={lambda}
              min={0.2}
              max={4}
              step={0.2}
              format={(v) => v.toFixed(1)}
              onChange={setLambda}
            />
            <Button className="w-full" disabled={loading} onClick={() => void run()}>
              {loading ? "Integrating…" : "Run Harmony"}
            </Button>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Tip: set <span className="font-mono">nclust</span> near the number
              of cell types. Too many clusters lets k-means isolate each
              donor-type blob, and Harmony2 then skips the ridge step.
            </p>
          </CardContent>
        </Card>

        <div className="space-y-6">
          {error ? (
            <Card className="border-destructive/40">
              <CardContent className="py-6 text-sm text-destructive">
                {error}
              </CardContent>
            </Card>
          ) : null}

          {!result && !error ? (
            <Card>
              <CardContent className="py-16 text-center text-sm text-muted-foreground">
                {loading
                  ? "Soft-clustering cells and fitting the ridge correction…"
                  : "Run Harmony to compare the raw PCA with the integrated embedding."}
              </CardContent>
            </Card>
          ) : null}

          {result ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-sm text-muted-foreground">
                  {result.beforeEmbeddings.length} cells · {result.nClusters}{" "}
                  clusters · {result.nIter} Harmony rounds
                </p>
                <Tabs value={colorBy} onValueChange={setColorBy}>
                  <TabsList>
                    <TabsTrigger value="batch">Color by donor</TabsTrigger>
                    <TabsTrigger value="cellType">Color by type</TabsTrigger>
                  </TabsList>
                  <TabsContent value="batch" />
                  <TabsContent value="cellType" />
                </Tabs>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <ScatterPlot
                  points={result.beforeEmbeddings}
                  labels={labels}
                  title="Before"
                  subtitle="PCA with a planted donor shift"
                />
                <ScatterPlot
                  points={result.afterEmbeddings}
                  labels={labels}
                  title="After harmorust"
                  subtitle="Same cells, Harmony-corrected PCs"
                />
              </div>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <MetricCard
                  label="Batch separation"
                  before={result.before.batchSeparation}
                  after={result.after.batchSeparation}
                  hint="should fall"
                />
                <MetricCard
                  label="Batch LISI"
                  before={result.before.batchLisi}
                  after={result.after.batchLisi}
                  hint="should rise toward the donor count"
                  invert
                />
                <MetricCard
                  label="Type separation"
                  before={result.before.typeSeparation}
                  after={result.after.typeSeparation}
                  hint="should hold"
                />
                <MetricCard
                  label="Type LISI"
                  before={result.before.typeLisi}
                  after={result.after.typeLisi}
                  hint="should stay near 1"
                />
              </div>
            </>
          ) : null}

          <Card>
            <CardHeader>
              <CardTitle>How this sits next to seurust</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
              <p>
                seurust replaces Seurat&apos;s C++ kernels (
                <span className="font-mono text-xs">LogNorm</span>, sparse
                scaling, SNN, integration helpers) with extendr/Rust. Harmony is
                the next native kernel in that pipeline: after{" "}
                <span className="font-mono text-xs">RunPCA</span>,{" "}
                <span className="font-mono text-xs">RunHarmony</span> only sees
                the cell embedding and a batch column.
              </p>
              <pre className="overflow-x-auto rounded-lg bg-muted px-3 py-3 font-mono text-[11px] leading-5 text-foreground">
{`# seurust handles NormalizeData / ScaleData / FindNeighbors
seurat <- RunPCA(seurat)
# harmorust drop-in, same signature as harmony::RunHarmony
seurat <- harmorust::RunHarmony(seurat, group.by.vars = "donor")
seurat <- RunUMAP(seurat, reduction = "harmony")`}
              </pre>
              <p>
                The Rust crate is the source of truth (
                <span className="font-mono text-xs">cargo test</span> asserts
                the donor shift disappears and cell types do not). This page
                runs the same algorithm so you can see it. An extendr wrapper
                lives under <span className="font-mono text-xs">r/harmorust</span>{" "}
                for the Seurat object path.
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <Label className="text-sm">{label}</Label>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {format(value)}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(next) => {
          const raw = Array.isArray(next) ? next[0] : next;
          if (typeof raw === "number") onChange(raw);
        }}
      />
    </div>
  );
}
