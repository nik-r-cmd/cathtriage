import { create } from "zustand";
import { GROUPS, mockCases } from "./mock";
// DATA CONTRACT: public/cases.json overrides the mock (see mock.js for the full shape).
export const useStore = create((set) => ({
  groups: GROUPS, cases: mockCases, source: "MOCK DATA", meta: { alpha: 0.1, model: "mock" },
  caseId: mockCases[0].id, step: 0, selected: null, hover: null, opacity: 0.5, clip: 0.9, playing: false, fly: null,
  load: (j) => set({ groups: j.groups, cases: j.cases, meta: j.meta || {}, source: "MODEL OUTPUT", caseId: j.cases[0].id, step: 0, selected: null }),
  setCase: (id) => set({ caseId: id, step: 0, selected: null, playing: false }),
  setStep: (step) => set({ step }), select: (selected) => set({ selected }), setHover: (hover) => set({ hover }),
  set,
}));
export function useCurrent() {
  const { cases, caseId, step } = useStore();
  const c = cases.find((x) => x.id === caseId) || cases[0];
  return { c, s: c.steps[Math.min(step, c.steps.length - 1)], last: c.steps[c.steps.length - 1] };
}
