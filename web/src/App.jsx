import { useEffect } from "react";
import Scene from "./Scene";
import Dashboard from "./Dashboard";
import { useStore } from "./store";

export default function App() {
  const load = useStore((s) => s.load);
  useEffect(() => {                                   // real model output replaces the mock when present
    fetch("./cases.json").then((r) => (r.ok ? r.json() : null)).then((j) => j && j.cases?.length && load(j)).catch(() => {});
  }, [load]);
  return (
    <div className="app">
      <div className="disclaimer" role="alert">
        DECISION SUPPORT / EDUCATIONAL USE ONLY - not a diagnosis and not a substitute for formal diagnostic imaging or clinical judgement.
        Heart and vessels are a schematic risk map, not patient anatomy. Retrospective single-centre data.
      </div>
      <main><div className="canvas"><Scene /></div><Dashboard /></main>
    </div>
  );
}
