import { useEffect } from "react";
import { useStore, useCurrent } from "./store";

const LABEL = { pos: "Likely stenosis", neg: "Unlikely", ambiguous: "Uncertain", empty: "Uncertain" };
const COL = { pos: "#e74c3c", neg: "#2ecc71", ambiguous: "#f1c40f", empty: "#f1c40f" };
const mix = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
const ramp = (p) => `rgb(${(p < 0.5 ? mix([46, 204, 113], [241, 196, 15], p * 2) : mix([241, 196, 15], [231, 76, 60], (p - 0.5) * 2)).join(",")})`;

const Ring = ({ p, color, size = 84 }) => {
  const r = size / 2 - 7, c = 2 * Math.PI * r;
  return (
    <svg width={size} height={size}>
      <circle cx={size / 2} cy={size / 2} r={r} stroke="#1f2937" strokeWidth="7" fill="none" />
      <circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth="7" fill="none" strokeLinecap="round"
        strokeDasharray={`${c * p} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: "stroke-dasharray .5s" }} />
      <text x="50%" y="52%" textAnchor="middle" dominantBaseline="middle" fill="#e6edf3" fontSize={size / 4} fontWeight="700">{Math.round(p * 100)}%</text>
    </svg>
  );
};
const SetChips = ({ set }) => (
  <span className="chips2">
    <b className={set === "neg" || set === "ambiguous" ? "on" : ""}>Normal</b>
    <b className={set === "pos" || set === "ambiguous" ? "on" : ""}>Stenosis</b>
  </span>
);

// AHA 17-segment polar map; standard territory assignment (individual anatomy varies)
const TERR = { LAD: [1, 2, 7, 8, 13, 14, 17], RCA: [3, 4, 9, 10, 15], LCX: [5, 6, 11, 12, 16] };
const vesselOf = (n) => Object.keys(TERR).find((k) => TERR[k].includes(n));
const P = (r, a) => [100 - r * Math.sin((a * Math.PI) / 180), 100 - r * Math.cos((a * Math.PI) / 180)];
const arc = (r0, r1, a0, a1) => { const [x1, y1] = P(r1, a0), [x2, y2] = P(r1, a1), [x3, y3] = P(r0, a1), [x4, y4] = P(r0, a0);
  return `M${x1},${y1}A${r1},${r1} 0 0 0 ${x2},${y2}L${x3},${y3}A${r0},${r0} 0 0 1 ${x4},${y4}Z`; };
function Bullseye({ probs }) {
  const { selected, hover, select, setHover } = useStore();
  const segs = [];
  for (let k = 0; k < 6; k++) { segs.push([k + 1, arc(62, 92, k * 60 - 30, k * 60 + 30)]); segs.push([k + 7, arc(38, 62, k * 60 - 30, k * 60 + 30)]); }
  for (let k = 0; k < 4; k++) segs.push([k + 13, arc(16, 38, k * 90 - 45, k * 90 + 45)]);
  return (
    <svg viewBox="0 0 200 200" className="bull">
      {segs.map(([n, d]) => { const v = vesselOf(n), on = (selected || hover) === v;
        return <path key={n} d={d} fill={ramp(probs[v])} opacity={on || !(selected || hover) ? 1 : 0.3} stroke="#0d1117" strokeWidth="1.5"
          onMouseEnter={() => setHover(v)} onMouseLeave={() => setHover(null)} onClick={() => select(selected === v ? null : v)} style={{ cursor: "pointer" }} />; })}
      {(() => { const v = "LAD", on = (selected || hover) === v;
        return <circle cx="100" cy="100" r="16" fill={ramp(probs[v])} opacity={on || !(selected || hover) ? 1 : 0.3} stroke="#0d1117" strokeWidth="1.5"
          onMouseEnter={() => setHover(v)} onMouseLeave={() => setHover(null)} onClick={() => select(selected === v ? null : v)} style={{ cursor: "pointer" }} />; })()}
      {segs.map(([n]) => { const k = n <= 6 ? n - 1 : n <= 12 ? n - 7 : n - 13, r = n <= 6 ? 77 : n <= 12 ? 50 : 27, a = n <= 12 ? k * 60 : k * 90, [x, y] = P(r, a);
        return <text key={"t" + n} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize="9" fill="#0d1117" pointerEvents="none">{n}</text>; })}
      <text x="100" y="100" textAnchor="middle" dominantBaseline="middle" fontSize="9" fill="#0d1117" pointerEvents="none">17</text>
    </svg>
  );
}

function Explain({ items }) {
  const m = Math.max(0.001, ...items.map((i) => Math.abs(i.c)));
  return items.map((i) => (
    <div key={i.f} className="shaprow">
      <span className="fname">{i.f}<em>{i.v}</em></span>
      <div className="diverge"><div style={{ width: `${(Math.abs(i.c) / m) * 50}%`, [i.c >= 0 ? "left" : "right"]: "50%", background: i.c >= 0 ? "#e74c3c" : "#3498db" }} /></div>
      <span className="cval">{i.c >= 0 ? "+" : ""}{i.c.toFixed(2)}</span>
    </div>
  ));
}

export default function Dashboard() {
  const { cases, caseId, step, selected, groups, source, meta, playing, setCase, setStep, select, set } = useStore();
  const { c, s } = useCurrent();
  const total = groups.reduce((a, g) => a + g.cost, 0), gl = Object.fromEntries(groups.map((g) => [g.id, g]));
  const focus = selected || "CAD", T = c.steps.length - 1;
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => useStore.setState((st) => (st.step >= T ? { playing: false } : { step: st.step + 1 })), 1100);
    return () => clearInterval(id);
  }, [playing, T]);
  const unc = ["LAD", "LCX", "RCA"].filter((v) => s.sets[v] === "ambiguous" || s.sets[v] === "empty").length;
  const headline = s.sets.CAD === "pos" ? "High likelihood of significant stenosis" : s.sets.CAD === "neg" ? "Significant stenosis unlikely" : "Not yet decisive";
  return (
    <aside className="panel">
      <div className="row between"><h1>Cath<span>Triage</span></h1><span className={`badge ${source === "MOCK DATA" ? "warn" : "ok"}`}>{source}</span></div>
      <select value={caseId} onChange={(e) => setCase(e.target.value)}>{cases.map((x) => <option key={x.id}>{x.id}</option>)}</select>

      <section className="card hero">
        <Ring p={s.p.CAD} color={COL[s.sets.CAD]} size={104} />
        <div><small>Overall CAD</small><div className="headline" style={{ color: COL[s.sets.CAD] }}>{headline}</div>
          <small>{unc} of 3 vessels uncertain at alpha={meta.alpha ?? 0.1}</small></div>
      </section>

      <section className="card">
        {["LAD", "LCX", "RCA"].map((v) => (
          <div key={v} className={`vcard ${selected === v ? "sel" : ""}`} onClick={() => select(selected === v ? null : v)}>
            <Ring p={s.p[v]} color={COL[s.sets[v]]} size={58} />
            <div><b>{v}</b> <small>{LABEL[s.sets[v]]}</small><SetChips set={s.sets[v]} /></div>
          </div>
        ))}
        <small>Narrowing is vessel-level. Its position along the artery is not predicted.</small>
      </section>

      <section className="card">
        <div className="row between"><b>Tests ordered</b><span>cost {s.cost.toFixed(1)} / {total.toFixed(1)}</span></div>
        <div className="bar"><div style={{ width: `${(s.cost / total) * 100}%` }} /></div>
        <div className="steps">{c.steps.slice(0, T).map((st, i) => (
          <button key={i} className={i < step ? "done" : i === step ? "active" : ""} onClick={() => setStep(i)} title={`cost ${gl[st.next]?.cost}`}>{gl[st.next]?.label ?? st.next}</button>))}</div>
        <div className="row"><button className="play" onClick={() => set({ playing: !playing, step: step >= T ? 0 : step })}>{playing ? "Pause" : "Replay"}</button>
          <small>{s.next ? <>Next: <b>{gl[s.next]?.label}</b></> : "Done"} - saved {(total - s.cost).toFixed(1)} units</small></div>
      </section>

      <section className="card"><b>Perfusion territories (AHA 17-segment)</b>
        <Bullseye probs={s.p} /><small>Hover or click a segment to highlight its artery in 3D.</small></section>

      <section className="card"><b>Why: {focus === "CAD" ? "overall CAD" : focus}</b><Explain items={s.shap[focus] || []} />
        <small>Red raises risk, blue lowers it. Only tests already ordered are shown.</small></section>
      {c.truth && <small className="muted">Demo ground truth: LAD {c.truth.LAD} / LCX {c.truth.LCX} / RCA {c.truth.RCA}</small>}
    </aside>
  );
}
