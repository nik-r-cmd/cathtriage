// Deterministic mock cases so the UI can be built before real model output exists.
export const GROUPS = [
  { id: "demographics", label: "Demographics", cost: 0, features: ["Age", "Sex", "BMI"] },
  { id: "history", label: "History / risk factors", cost: 0, features: ["DM", "HTN", "Current Smoker", "FH", "DLP"] },
  { id: "symptoms", label: "Symptoms", cost: 0.5, features: ["Typical Chest Pain", "Exertional CP", "Dyspnea"] },
  { id: "exam_vitals", label: "Exam & vitals", cost: 1, features: ["BP", "PR", "Edema"] },
  { id: "ecg", label: "ECG", cost: 3, features: ["St Depression", "Tinversion", "Q Wave", "LVH"] },
  { id: "lab_lipid", label: "Lipid panel", cost: 4, features: ["LDL", "HDL", "TG"] },
  { id: "lab_glucose_renal", label: "Glucose / renal", cost: 4, features: ["FBS", "CR", "HB"] },
  { id: "lab_cbc_inflam", label: "CBC / inflammation", cost: 4, features: ["ESR", "WBC", "Neut", "Lymph"] },
  { id: "echo", label: "Echocardiogram", cost: 8, features: ["EF-TTE", "Region RWMA"] },
];

let seed = 7;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const setOf = (p) => (p >= 0.8 ? "pos" : p <= 0.2 ? "neg" : "ambiguous");

function makeCase(i) {
  const truth = { LAD: rnd() < 0.58 ? 1 : 0, LCX: rnd() < 0.4 ? 1 : 0, RCA: rnd() < 0.38 ? 1 : 0 };
  const paid = GROUPS.filter((g) => g.cost > 0).sort(() => rnd() - 0.5);
  const free = GROUPS.filter((g) => g.cost === 0).map((g) => g.id);
  const feats = {}; GROUPS.forEach((g) => g.features.forEach((f) => (feats[f] = +(rnd() * 2 - 1).toFixed(2))));
  const steps = []; let revealed = [...free], cost = 0;
  for (let t = 0; t <= paid.length; t++) {
    const f = Math.pow(t / paid.length, 0.8) * 0.8;
    const p = {};
    ["LAD", "LCX", "RCA"].forEach((v) => {
      const prior = { LAD: 0.58, LCX: 0.4, RCA: 0.38 }[v];
      p[v] = Math.min(0.98, Math.max(0.02, prior + ((truth[v] ? 0.92 : 0.08) - prior) * f + (rnd() - 0.5) * 0.12));
    });
    p.CAD = 1 - (1 - p.LAD) * (1 - p.LCX) * (1 - p.RCA);
    const shap = {};
    ["LAD", "LCX", "RCA", "CAD"].forEach((v) => {
      const pool = GROUPS.filter((g) => revealed.includes(g.id)).flatMap((g) => g.features);
      shap[v] = pool.map((ft) => ({ f: ft, v: feats[ft], c: +((rnd() - 0.5) * 0.5).toFixed(3) }))
        .sort((a, b) => Math.abs(b.c) - Math.abs(a.c)).slice(0, 6);
    });
    steps.push({ cost, revealed: [...revealed], next: paid[t] ? paid[t].id : null, p,
                 sets: { LAD: setOf(p.LAD), LCX: setOf(p.LCX), RCA: setOf(p.RCA), CAD: setOf(p.CAD) }, shap });
    if (paid[t]) { revealed = [...revealed, paid[t].id]; cost += paid[t].cost; }
  }
  return { id: `MOCK-${String(i + 1).padStart(3, "0")}`, truth, steps };
}
export const mockCases = Array.from({ length: 12 }, (_, i) => makeCase(i));
