import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";

function tubeGeo(curve, T = 72, R = 12) {
  const fr = curve.computeFrenetFrames(T, false), pos = [], dir = [], tt = [], aa = [], idx = [];
  for (let i = 0; i <= T; i++) {
    const c = curve.getPoint(i / T);
    for (let j = 0; j <= R; j++) {
      const a = (j / R) * Math.PI * 2, d = fr.normals[i].clone().multiplyScalar(Math.cos(a)).addScaledVector(fr.binormals[i], Math.sin(a));
      pos.push(c.x, c.y, c.z); dir.push(d.x, d.y, d.z); tt.push(i / T); aa.push(j / R);
      if (i < T && j < R) { const q = i * (R + 1) + j; idx.push(q, q + R + 1, q + 1, q + 1, q + R + 1, q + R + 2); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute("aDir", new THREE.Float32BufferAttribute(dir, 3));
  g.setAttribute("aT", new THREE.Float32BufferAttribute(tt, 1)); g.setAttribute("aA", new THREE.Float32BufferAttribute(aa, 1));
  g.setIndex(idx); return g;
}
// Narrowing happens on the GPU (smooth tween). Its position along the vessel is NOT predicted (vessel-level labels only).
const VERT = `uniform float uR,uSev; attribute vec3 aDir; attribute float aT,aA; varying float vT,vA; varying vec3 vN,vV;
void main(){ float env=exp(-pow((aT-.45)/.28,2.)); float r=uR*mix(1.,.55,aT)*(1.-uSev*.78*env);
 vec4 mv=modelViewMatrix*vec4(position+aDir*r,1.); vN=normalize(normalMatrix*aDir); vV=-mv.xyz; vT=aT; vA=aA; gl_Position=projectionMatrix*mv; }`;
const FRAG = `uniform float uProb,uTime,uAmb,uSel,uDim; varying float vT,vA; varying vec3 vN,vV;
vec3 ramp(float p){ vec3 g=vec3(.18,.8,.45),y=vec3(.95,.77,.2),r=vec3(.91,.3,.24); return p<.5?mix(g,y,p*2.):mix(y,r,(p-.5)*2.); }
void main(){ vec3 n=normalize(vN),v=normalize(vV); float fr=pow(1.-max(dot(n,v),0.),2.5); vec3 b=ramp(uProb);
 float flow=smoothstep(.55,1.,sin((vT*28.-uTime*(1.-.7*uProb)*3.)*3.14159));
 float hatch=uAmb>.5?step(.5,fract(vT*60.+vA*2.))*(.5+.5*sin(uTime*4.)):0.;
 vec3 col=b*(.35+.5*max(dot(n,normalize(vec3(.4,.7,.6))),0.))+b*flow*.5+fr*.35+hatch*.25+b*uSel*.5; gl_FragColor=vec4(col*mix(1.,.35,uDim),1.); }`;

export default function Vessel({ seg, prob, amb, selected, dim, onSelect, onHover }) {
  const m = useRef(), cur = useRef(prob), geo = useMemo(() => tubeGeo(seg.curve), [seg]);
  const uniforms = useMemo(() => ({ uR: { value: seg.r }, uSev: { value: 0 }, uProb: { value: prob }, uTime: { value: 0 },
    uAmb: { value: 0 }, uSel: { value: 0 }, uDim: { value: 0 } }), [seg]);
  useFrame(({ clock }) => {
    const u = m.current.uniforms; cur.current += (prob - cur.current) * 0.07;
    u.uProb.value = cur.current; u.uSev.value = cur.current * (seg.main ? 1 : 0.45); u.uTime.value = clock.elapsedTime;
    u.uAmb.value = amb ? 1 : 0; u.uSel.value += ((selected ? 1 : 0) - u.uSel.value) * 0.15; u.uDim.value += ((dim ? 1 : 0) - u.uDim.value) * 0.15;
  });
  return (
    <mesh geometry={geo} onClick={(e) => { e.stopPropagation(); onSelect(seg.v); }}
      onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = "pointer"; onHover(seg.v); }}
      onPointerOut={() => { document.body.style.cursor = ""; onHover(null); }}>
      <shaderMaterial ref={m} uniforms={uniforms} vertexShader={VERT} fragmentShader={FRAG} />
    </mesh>
  );
}
