"use client";

const BATCH_COLORS: Record<string, string> = {
  donor_A: "#c45c26",
  donor_B: "#1f6f8b",
  donor_C: "#5b4b8a",
  donor_D: "#2f6b3a",
};

const TYPE_COLORS: Record<string, string> = {
  "T cell": "#c45c26",
  "B cell": "#1f6f8b",
  Monocyte: "#5b4b8a",
  "NK cell": "#2f6b3a",
  DC: "#8a4b2f",
};

const FALLBACK = [
  "#c45c26",
  "#1f6f8b",
  "#5b4b8a",
  "#2f6b3a",
  "#8a4b2f",
  "#6b2f4b",
];

type Props = {
  points: number[][];
  labels: string[];
  title: string;
  subtitle: string;
};

export function ScatterPlot({ points, labels, title, subtitle }: Props) {
  if (!points.length) {
    return (
      <div className="flex aspect-square items-center justify-center rounded-xl border border-dashed bg-muted/30 text-sm text-muted-foreground">
        No embeddings yet
      </div>
    );
  }

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padX = (maxX - minX || 1) * 0.12;
  const padY = (maxY - minY || 1) * 0.12;
  const unique = Array.from(new Set(labels));

  const xOf = (v: number) => ((v - (minX - padX)) / (maxX - minX + 2 * padX)) * 100;
  const yOf = (v: number) =>
    (1 - (v - (minY - padY)) / (maxY - minY + 2 * padY)) * 100;

  return (
    <figure className="space-y-3">
      <div>
        <h3 className="font-heading text-base font-medium">{title}</h3>
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      </div>
      <div className="relative aspect-square overflow-hidden rounded-xl border bg-[color:var(--card)]">
        <svg viewBox="0 0 100 100" className="h-full w-full" role="img" aria-label={title}>
          <rect width="100" height="100" fill="transparent" />
          {points.map((p, i) => {
            const color =
              BATCH_COLORS[labels[i]] ??
              TYPE_COLORS[labels[i]] ??
              FALLBACK[unique.indexOf(labels[i]) % FALLBACK.length];
            return (
              <circle
                key={`${i}-${p[0]}`}
                cx={xOf(p[0])}
                cy={yOf(p[1])}
                r={1.15}
                fill={color}
                fillOpacity={0.88}
              />
            );
          })}
        </svg>
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {unique.map((label, i) => (
          <li key={label} className="inline-flex items-center gap-1.5">
            <span
              className="size-2 rounded-full"
              style={{
                background:
                  BATCH_COLORS[label] ??
                  TYPE_COLORS[label] ??
                  FALLBACK[i % FALLBACK.length],
              }}
            />
            {label}
          </li>
        ))}
      </ul>
    </figure>
  );
}
