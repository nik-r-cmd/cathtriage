// One coronary vessel: lumen narrows with P(stenosis), colour ramps green->red,
// conformal-ambiguous vessels pulse. Schematic risk mapping, NOT patient anatomy.
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";

const GREEN = new THREE.Color("#2ecc71"), YELLOW = new THREE.Color("#f1c40f"), RED = new THREE.Color("#e74c3c");
const ramp = (p) => (p < 0.5 ? GREEN.clone().lerp(YELLOW, p / 0.5) : YELLOW.clone().lerp(RED, (p - 0.5) / 0.5));

function stenosedTube(curve, baseR, severity, center = 0.5, width = 0.12) {
  const TS = 64, RS = 12;
  const g = new THREE.TubeGeometry(curve, TS, baseR, RS, false);
  const pos = g.attributes.position, c = new THREE.Vector3(), v = new THREE.Vector3();
  for (let i = 0; i <= TS; i++) {
    curve.getPointAt(i / TS, c);
    const z = ((i / TS) - center) / width;
    const f = 1 - severity * 0.8 * Math.exp(-(z * z));
    for (let j = 0; j <= RS; j++) {
      const k = i * (RS + 1) + j;
      v.fromBufferAttribute(pos, k).sub(c).multiplyScalar(f).add(c);
      pos.setXYZ(k, v.x, v.y, v.z);
    }
  }
  g.computeVertexNormals();
  return g;
}

export default function Vessel({ name, points, prob, ambiguous, selected, onSelect, baseR = 0.035 }) {
  const mat = useRef();
  const curve = useMemo(() => new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p))), [points]);
  const geo = useMemo(() => stenosedTube(curve, baseR, prob), [curve, baseR, prob]);
  useFrame(({ clock }) => {
    if (!mat.current) return;
    mat.current.emissiveIntensity = ambiguous ? 0.35 + 0.35 * Math.sin(clock.elapsedTime * 4) : selected ? 0.5 : 0.08;
  });
  const col = ramp(prob);
  return (
    <mesh geometry={geo} onClick={(e) => { e.stopPropagation(); onSelect(name); }}>
      <meshStandardMaterial ref={mat} color={col} emissive={col} roughness={0.45} />
    </mesh>
  );
}

// Placeholder control points (tune visually against your heart mesh scale):
export const VESSELS = {
  LAD: [[0.0, 0.5, 0.45], [0.15, 0.2, 0.5], [0.25, -0.2, 0.45], [0.3, -0.55, 0.3]],
  LCX: [[0.0, 0.5, 0.45], [-0.3, 0.35, 0.3], [-0.45, 0.0, 0.1], [-0.4, -0.35, -0.1]],
  RCA: [[0.0, 0.55, 0.4], [0.35, 0.45, 0.3], [0.5, 0.0, 0.1], [0.35, -0.45, -0.1]],
};
