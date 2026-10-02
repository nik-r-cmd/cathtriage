import { create } from "zustand";
import { GROUPS, mockCases } from "./mock";

// DATA CONTRACT (see mock.js for a full example). public/cases.json overrides the mock:
// { meta:{alpha, model, ...}, groups:[{id,label,cost,features}], cases:[{id, truth?, steps:[{cost, revealed:[groupId], next, p:{LAD,LCX,RCA,CAD}, sets:{LAD:'pos|neg|ambiguous|empty',...}, shap:{LAD:[{f,v,c}],...}}]}] }
export const useStore = create((set, get) => ({
  groups: GROUPS, cases: mockCases, source: "MOCK DATA", meta: { alpha: 0.1, model: "mock" },
  caseId: mockCases[0].id, step: 0, selected: null,
  load: (json) => set({ groups: json.groups, cases: json.cases, meta: json.meta || {}, source: "MODEL OUTPUT",
                        caseId: json.cases[0].id, step: 0, selected: null }),
  setCase: (id) => set({ caseId: id, step: 0, selected: null }),
  setStep: (step) => set({ step }),
  select: (selected) => set({ selected }),
}));

export function useCurrent() {
  const { cases, caseId, step } = useStore();
  const c = cases.find((x) => x.id === caseId) || cases[0];
  return { c, s: c.steps[Math.min(step, c.steps.length - 1)], last: c.steps[c.steps.length - 1] };
}
