import { DemoApp } from "@/components/demo-app";
import { computeDemo, DEFAULT_PARAMS } from "@/lib/run-demo";

export default function Home() {
  const initial = computeDemo(DEFAULT_PARAMS);
  return <DemoApp initial={initial} />;
}
