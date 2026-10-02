import { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import Vessel, { VESSELS } from "./Vessel";
import { useStore, useCurrent } from "./store";

// SCHEMATIC heart (procedural placeholder). To use a real mesh: drop public/models/heart.glb,
// load it with drei's useGLTF, render it instead of <HeartBody/>, and re-tune VESSELS control points.
// Check the mesh licence (CC-BY etc.) and credit it in the README + UI footer.
function HeartBody() {
  return (
    <group>
      <mesh scale={[0.9, 1.1, 0.8]}><sphereGeometry args={[0.62, 48, 48]} /><meshStandardMaterial color="#7a2a36" roughness={0.65} transparent opacity={0.92} /></mesh>
      <mesh position={[-0.28, 0.62, -0.05]}><sphereGeometry args={[0.24, 24, 24]} /><meshStandardMaterial color="#6a2230" roughness={0.7} /></mesh>
      <mesh position={[0.3, 0.6, -0.05]}><sphereGeometry args={[0.22, 24, 24]} /><meshStandardMaterial color="#6a2230" roughness={0.7} /></mesh>
      <mesh position={[0.02, 0.95, -0.1]}><cylinderGeometry args={[0.12, 0.14, 0.5, 24]} /><meshStandardMaterial color="#9a3a48" roughness={0.6} /></mesh>
    </group>
  );
}

function BeatingHeart() {
  const g = useRef();
  const { c, s } = useCurrent();
  const selected = useStore((x) => x.selected);
  const select = useStore((x) => x.select);
  useFrame(({ clock }) => {
    const k = 1 + 0.035 * Math.pow(Math.max(0, Math.sin(clock.elapsedTime * 2 * Math.PI * 1.2)), 3);   // purely cosmetic
    g.current.scale.set(k, k, k);
  });
  return (
    <group ref={g} rotation={[0.1, -0.5, 0.05]}>
      <HeartBody />
      {["LAD", "LCX", "RCA"].map((v) => (
        <Vessel key={v + c.id} name={v} points={VESSELS[v]} prob={s.p[v]}
                ambiguous={s.sets[v] === "ambiguous" || s.sets[v] === "empty"}
                selected={selected === v} onSelect={select} />
      ))}
    </group>
  );
}

export default function Scene() {
  const select = useStore((x) => x.select);
  return (
    <Canvas camera={{ position: [0, 0.2, 2.6], fov: 45 }} dpr={[1, 1.5]} onPointerMissed={() => select(null)}>
      <ambientLight intensity={0.7} />
      <directionalLight position={[3, 4, 5]} intensity={1.1} />
      <directionalLight position={[-3, -2, -4]} intensity={0.4} />
      <BeatingHeart />
      <OrbitControls enablePan={false} minDistance={1.4} maxDistance={5} />
    </Canvas>
  );
}
