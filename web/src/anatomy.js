import * as THREE from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
// Schematic anatomy: x = patient's left, y = up, z = anterior. NOT patient-specific.
export function makeBody() {
  let g = new THREE.IcosahedronGeometry(1, 5);
  g.deleteAttribute("normal"); g.deleteAttribute("uv"); g = mergeVertices(g);
  const p = g.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const t = v.y < 0 ? 1 + 0.7 * v.y : 1;                       // taper to apex
    let x = v.x * 0.62 * t - 0.28 * Math.min(v.y, 0), y = v.y * 0.82, z = v.z * 0.52 * t;   // apex leans to patient's left
    const L = Math.hypot(0.4, -1.4), d = Math.abs((x - 0.05) * -1.4 - (y - 0.55) * 0.4) / L;
    const g1 = 0.05 * Math.exp(-((d / 0.06) ** 2)) * Math.min(1, Math.abs(z) * 4);           // interventricular groove
    const g2 = 0.045 * Math.exp(-(((y - (0.3 - 0.1 * x)) / 0.06) ** 2));                     // AV groove
    const k = 1 - g1 - g2 + 0.006 * Math.sin(x * 28) * Math.sin(y * 26 + 1) * Math.sin(z * 30);
    p.setXYZ(i, x * k, y * k, z * k);
  }
  g.computeVertexNormals();
  return g;
}
const snapper = (geo) => {
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide })), rc = new THREE.Raycaster();
  return (a) => { const d = new THREE.Vector3(...a).normalize(); rc.set(d.clone().multiplyScalar(3), d.clone().negate());
    const h = rc.intersectObject(m)[0]; return h ? h.point.clone().addScaledVector(d, 0.015) : new THREE.Vector3(...a); };
};
// anchors are directions from the heart centre; they are ray-snapped onto the surface
export const TREE = [
  { id: "LAD", v: "LAD", main: 1, r: .034, a: [[.05,.62,.45],[.2,.3,.75],[.3,-.1,.65],[.4,-.4,.5],[.45,-.7,.35],[.4,-.9,.1]] },
  { id: "D1", v: "LAD", p: "LAD", f: .3, r: .018, a: [[.5,.2,.6],[.65,0,.45]] },
  { id: "D2", v: "LAD", p: "LAD", f: .55, r: .016, a: [[.55,-.3,.5],[.65,-.5,.3]] },
  { id: "LCX", v: "LCX", main: 1, r: .03, a: [[.05,.62,.45],[.35,.45,.55],[.58,.35,.25],[.66,.2,-.15],[.55,.1,-.5],[.3,.05,-.7]] },
  { id: "OM1", v: "LCX", p: "LCX", f: .4, r: .017, a: [[.7,-.1,.1],[.62,-.45,-.1]] },
  { id: "OM2", v: "LCX", p: "LCX", f: .7, r: .015, a: [[.55,-.3,-.45],[.4,-.55,-.5]] },
  { id: "RCA", v: "RCA", main: 1, r: .032, a: [[-.05,.62,.45],[-.4,.35,.55],[-.6,.1,.3],[-.6,-.1,-.1],[-.4,-.15,-.5],[-.1,-.2,-.7]] },
  { id: "AM", v: "RCA", p: "RCA", f: .45, r: .017, a: [[-.55,-.2,.4],[-.4,-.6,.35]] },
  { id: "PDA", v: "RCA", p: "RCA", f: 1, r: .02, a: [[.05,-.4,-.7],[.2,-.75,-.4]] },
];
export function buildAnatomy() {
  const body = makeBody(), snap = snapper(body), segs = {};
  for (const s of TREE) {
    const pts = s.a.map(snap);
    if (s.p) pts.unshift(segs[s.p].curve.getPoint(s.f));
    segs[s.id] = { ...s, main: !!s.main, curve: new THREE.CatmullRomCurve3(pts, false, "centripetal") };
  }
  return { body, segs: Object.values(segs) };
}
const C = (pts) => new THREE.CatmullRomCurve3(pts.map((q) => new THREE.Vector3(...q)));
export const GREAT = [
  { c: C([[0,.5,0],[0,1,0],[-.2,1.3,-.05],[-.45,1.2,-.2],[-.5,.7,-.45]]), r: .13, col: "#b5485a" },   // aorta
  { c: C([[.2,.5,.35],[.25,.95,.3],[.1,1.2,.1]]), r: .1, col: "#4a7fc0" },                              // pulmonary trunk
  { c: C([[-.4,.6,-.05],[-.4,1.25,-.05]]), r: .08, col: "#4a7fc0" },                                    // SVC
];
