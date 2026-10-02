import { useStore, useCurrent } from "./store";

const LABEL = { pos: "Likely stenosis", neg: "Unlikely", ambiguous: "Uncertain", empty: "Uncertain" };
const COLOR = { pos: "#e74c3c", neg: "#2ecc71", ambiguous: "#f1c40f", empty: "#f1c40f" };

function Bar({ p, color }) {
  return <div className="bar"><div style={{ width: `${Math.round(p * 100)}%`, background: color }} /></div>;
}

function Explain({ items }) {
  const m = Math.max(0.001, ...items.map((i) => Math.abs(i.c)));
  return (
    <div className="shap">
      {items.map((i) => (
        <div key={i.f} className="shaprow">
          <span className="fname">{i.f} <em>{i.v}</em></span>
          <div className="diverge">
            <div style={{ width: `${(Math.abs(i.c) / m) * 50}%`, [i.c >= 0 ? "left" : "right"]: "50%",
                          background: i.c >= 0 ? "#e74c3c" : "#3498db" }} />
          </div>
          <span className="cval">{i.c >= 0 ? "+" : ""}{i.c.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const { cases, caseId, step, selected, groups, source, meta, setCase, setStep, select } = useStore();
  const { c, s, last } = useCurrent();
  const total = groups.reduce((a, g) => a + g.cost, 0);
  const gl = Object.fromEntries(groups.map((g) => [g.id, g]));
  const focus = selected || "CAD";
  return (
    <aside className="panel">
      <div className="row between">
        <h1>CathTriage</h1><span className={`badge ${source === "MOCK DATA" ? "warn" : "ok"}`}>{source}</span>
      </div>
      <label>Patient case
        <select value={caseId} onChange={(e) => setCase(e.target.value)}>{cases.map((x) => <option key={x.id}>{x.id}</option>)}</select>
      </label>

      <section className="card">
        <div className="row between"><strong>Overall CAD risk</strong><span style={{ color: COLOR[s.sets.CAD] }}>{LABEL[s.sets.CAD]}</span></div>
        <div className="big">{Math.round(s.p.CAD * 100)}%</div>
        <Bar p={s.p.CAD} color={COLOR[s.sets.CAD]} />
      </section>

      <section className="card">
        <strong>Vessel stenosis (click to inspect)</strong>
        {["LAD", "LCX", "RCA"].map((v) => (
          <div key={v} className={`vrow ${selected === v ? "sel" : ""}`} onClick={() => select(selected === v ? null : v)}>
            <span className="vname">{v}</span><Bar p={s.p[v]} color={COLOR[s.sets[v]]} />
            <span className="vp">{Math.round(s.p[v] * 100)}%</span>
            <span className="tag" style={{ borderColor: COLOR[s.sets[v]], color: COLOR[s.sets[v]] }}>{LABEL[s.sets[v]]}</span>
          </div>
        ))}
        <small>Uncertain = the conformal set (alpha={meta.alpha ?? 0.1}) contains both outcomes. Colour = predicted probability.</small>
      </section>

      <section className="card">
        <div className="row between"><strong>Tests ordered</strong><span>cost {s.cost.toFixed(1)} / {total.toFixed(1)}</span></div>
        <input type="range" min={0} max={c.steps.length - 1} value={step} onChange={(e) => setStep(+e.target.value)} />
        <div className="chips">{s.revealed.map((id) => <span key={id} className="chip">{gl[id]?.label ?? id}</span>)}</div>
        <small>{s.next ? <>Suggested next test: <b>{gl[s.next]?.label ?? s.next}</b></> : "All tests used."}
          {" "}Saved vs. everything: {(total - s.cost).toFixed(1)} units.</small>
      </section>

      <section className="card">
        <strong>Why: {focus === "CAD" ? "overall CAD" : focus} (top drivers)</strong>
        <Explain items={s.shap[focus] || []} />
        <small>Red raises risk, blue lowers it. Only tests already ordered are shown.</small>
      </section>
      {c.truth && <small className="muted">Demo ground truth (hidden in real use): LAD {c.truth.LAD} / LCX {c.truth.LCX} / RCA {c.truth.RCA}</small>}
    </aside>
  );
}
