import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { CameraControls, Html } from "@react-three/drei";
import { XR, createXRStore, useXR } from "@react-three/xr";
import Vessel from "./Vessel";
import { buildAnatomy, GREAT } from "./anatomy";
import { useStore, useCurrent } from "./store";

const xr = createXRStore();
const VIEWS = {
  front: [0, .15, 3.1, .05, -.05, 0], left: [3, .2, .6, .2, 0, 0], back: [.2, .2, -3.1, 0, 0, 0],
  LAD: [1.1, .1, 2.4, .3, -.2, .5], LCX: [2.5, .4, .4, .55, .2, -.05], RCA: [-1.9, .2, 2, -.4, 0, .2],
};
const BODY_V = `varying vec3 vP,vN,vV; void main(){ vP=position; vN=normalize(normalMatrix*normal); vec4 mv=modelViewMatrix*vec4(position,1.); vV=-mv.xyz; gl_Position=projectionMatrix*mv; }`;
const BODY_F = `uniform float uO,uClip,uTime; varying vec3 vP,vN,vV;
void main(){ if(vP.z>uClip) discard; vec3 n=normalize(vN),v=normalize(vV); float f=pow(1.-abs(dot(n,v)),2.);
 vec3 c=mix(vec3(.3,.07,.13),vec3(.95,.35,.45),f)+.05*sin(vP.y*90.-uTime*2.); gl_FragColor=vec4(c,uO*(.3+.7*f)); }`;

function Heart() {
  const { body, segs } = useMemo(buildAnatomy, []);
  const { c, s } = useCurrent();
  const { selected, hover, opacity, clip, select, setHover } = useStore();
  const g = useRef(), bm = useRef(), inXR = useXR((x) => x.session != null);
  const bu = useMemo(() => ({ uO: { value: 0.5 }, uClip: { value: 1 }, uTime: { value: 0 } }), []);
  useFrame(({ clock }) => {
    const k = 1 + 0.03 * Math.pow(Math.max(0, Math.sin(clock.elapsedTime * 7.5)), 3);   // cosmetic beat, not physiology
    g.current.scale.setScalar(k * (inXR ? 0.35 : 1));
    bm.current.uniforms.uO.value = opacity; bm.current.uniforms.uClip.value = clip * 1.2 - 0.2; bm.current.uniforms.uTime.value = clock.elapsedTime;
  });
  const focus = selected || hover;
  return (
    <group ref={g} position={inXR ? [0, 1.3, -0.8] : [0, 0, 0]}>
      <mesh geometry={body} renderOrder={2}>
        <shaderMaterial ref={bm} uniforms={bu} vertexShader={BODY_V} fragmentShader={BODY_F} transparent depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      {GREAT.map((x, i) => (
        <mesh key={i}><tubeGeometry args={[x.c, 40, x.r, 16]} /><meshStandardMaterial color={x.col} transparent opacity={0.55} roughness={0.5} /></mesh>
      ))}
      {segs.map((sg) => (
        <Vessel key={sg.id + c.id} seg={sg} prob={s.p[sg.v]} amb={s.sets[sg.v] === "ambiguous" || s.sets[sg.v] === "empty"}
          selected={focus === sg.v} dim={!!focus && focus !== sg.v} onSelect={(v) => select(selected === v ? null : v)} onHover={setHover} />
      ))}
      {segs.filter((x) => x.main).map((sg) => (
        <Html key={"t" + sg.id} position={sg.curve.getPoint(0.5)} center style={{ pointerEvents: "none" }}>
          <div className="tag3d">{sg.v} {Math.round(s.p[sg.v] * 100)}%</div>
        </Html>
      ))}
    </group>
  );
}

function Rig() {
  const ref = useRef(), selected = useStore((x) => x.selected), inXR = useXR((x) => x.session != null);
  useEffect(() => { useStore.setState({ fly: (k) => ref.current?.setLookAt(...VIEWS[k], true) }); }, []);
  useEffect(() => { ref.current?.setLookAt(...VIEWS[selected || "front"], true); }, [selected]);
  return inXR ? null : <CameraControls ref={ref} minDistance={1.3} maxDistance={6} smoothTime={0.25} />;
}

export default function Scene() {
  const { opacity, clip, set, select, fly } = useStore();
  const [sup, setSup] = useState({ vr: false, ar: false });
  useEffect(() => {
    if (!navigator.xr) return;
    Promise.all(["immersive-vr", "immersive-ar"].map((m) => navigator.xr.isSessionSupported(m).catch(() => false))).then(([vr, ar]) => setSup({ vr, ar }));
  }, []);
  return (
    <div className="stage">
      <Canvas camera={{ position: [0, 0.15, 3.1], fov: 42 }} dpr={[1, 1.5]} onPointerMissed={() => select(null)}>
        <XR store={xr}>
          <ambientLight intensity={0.8} /><directionalLight position={[3, 4, 5]} intensity={1.1} /><directionalLight position={[-3, -2, -4]} intensity={0.4} />
          <Heart /><Rig />
        </XR>
      </Canvas>
      <div className="toolbar">
        {["front", "left", "back"].map((k) => <button key={k} onClick={() => { select(null); fly?.(k); }}>{k}</button>)}
        <label>Body<input type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(e) => set({ opacity: +e.target.value })} /></label>
        <label>Cut<input type="range" min="0" max="1" step="0.05" value={clip} onChange={(e) => set({ clip: +e.target.value })} /></label>
        <button disabled={!sup.vr} onClick={() => xr.enterVR()} title={sup.vr ? "" : "WebXR VR not available on this device/browser"}>VR</button>
        <button disabled={!sup.ar} onClick={() => xr.enterAR()} title={sup.ar ? "" : "WebXR AR not available on this device/browser"}>AR</button>
      </div>
      <div className="legend"><i style={{ background: "#2ecc71" }} />low <i style={{ background: "#f1c40f" }} />mid <i style={{ background: "#e74c3c" }} />high risk<span>Hatched pulse = uncertain (conformal)</span></div>
    </div>
  );
}
