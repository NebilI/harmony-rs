import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type Props = {
  label: string;
  before: number;
  after: number;
  hint: string;
  invert?: boolean;
};

export function MetricCard({ label, before, after, hint, invert }: Props) {
  const improved = invert ? after > before : after < before;
  const delta = after - before;
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="text-xs text-muted-foreground">Before</p>
            <p className="font-mono text-lg tabular-nums">{before.toFixed(2)}</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-muted-foreground">After</p>
            <p className="font-mono text-lg tabular-nums">{after.toFixed(2)}</p>
          </div>
        </div>
        <p className={`text-xs ${improved ? "text-emerald-700" : "text-amber-700"}`}>
          {delta >= 0 ? "+" : ""}
          {delta.toFixed(2)} · {hint}
        </p>
      </CardContent>
    </Card>
  );
}
