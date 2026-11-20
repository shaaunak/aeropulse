import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';
import type { Group } from 'three';

export type ModelMode = 'ASSEMBLED' | 'XRAY' | 'EXPLODED';
export type EnginePart = 'FAN' | 'COMPRESSOR' | 'CASING' | 'TURBINE' | 'NOZZLE' | null;

type Props = {
  mode: ModelMode;
  speed: number;
  paused: boolean;
  reducedMotion: boolean;
  selectedPart?: EnginePart;
  onSelectPart?: (part: EnginePart) => void;
};

function BladedStage({
  count,
  radius,
  z,
  hub,
  rotationSpeed,
  speed,
  paused,
  reducedMotion,
  color,
  highlighted = false,
}: {
  count: number;
  radius: number;
  z: number;
  hub: number;
  rotationSpeed: number;
  speed: number;
  paused: boolean;
  reducedMotion: boolean;
  color: string;
  highlighted?: boolean;
}) {
  const rotorRef = useRef<Group>(null);
  const shape = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-0.10, hub);
    s.bezierCurveTo(-0.19, hub + radius * 0.28, -0.23, hub + radius * 0.66, 0.08, hub + radius);
    s.bezierCurveTo(0.17, hub + radius * 0.93, 0.13, hub + radius * 0.77, 0.10, hub + radius * 0.64);
    s.bezierCurveTo(0.07, hub + radius * 0.32, 0.08, hub + radius * 0.12, 0.04, hub);
    s.closePath();
    return new THREE.ExtrudeGeometry(s, {
      depth: 0.065,
      bevelEnabled: true,
      bevelSegments: 1,
      bevelSize: 0.012,
      bevelThickness: 0.01,
      steps: 1,
      curveSegments: 8,
    });
  }, [hub, radius]);

  useFrame((_, dt) => {
    if (rotorRef.current && !paused && !reducedMotion) {
      rotorRef.current.rotation.z += Math.min(dt, 0.06) * speed * rotationSpeed;
    }
  });

  const stageColor = highlighted ? '#FF5E39' : color;
  const stageRoughness = highlighted ? 0.15 : 0.28;

  return (
    <group ref={rotorRef} position={[0, 0, z]}>
      {Array.from({ length: count }, (_, i) => (
        <mesh
          key={i}
          geometry={shape}
          rotation={[0, 0, (i / count) * Math.PI * 2]}
          castShadow
        >
          <meshStandardMaterial
            color={stageColor}
            roughness={stageRoughness}
            metalness={0.92}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
      <mesh rotation={[Math.PI / 2, 0, 0]} castShadow>
        <cylinderGeometry args={[hub + 0.035, hub + 0.035, 0.20, 48]} />
        <meshStandardMaterial color={highlighted ? '#FFA085' : '#798681'} metalness={0.94} roughness={0.27} />
      </mesh>
    </group>
  );
}

function HotspotPin({
  position,
  active,
  onClick,
}: {
  position: [number, number, number];
  label?: string;
  active: boolean;
  onClick: () => void;
}) {
  const pinRef = useRef<Group>(null);
  useFrame((state) => {
    if (pinRef.current) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.08;
      pinRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <group ref={pinRef} position={position} onClick={(e) => { e.stopPropagation(); onClick(); }}>
      <mesh>
        <sphereGeometry args={[0.075, 16, 16]} />
        <meshBasicMaterial color={active ? '#FF5E39' : '#F3F1EB'} />
      </mesh>
      <mesh>
        <ringGeometry args={[0.09, 0.13, 24]} />
        <meshBasicMaterial color={active ? '#FF5E39' : '#8C918B'} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function Turbine({ mode, speed, paused, reducedMotion, selectedPart, onSelectPart }: Props) {
  const exploded = mode === 'EXPLODED';
  const xray = mode === 'XRAY';

  // Physical separation offsets in EXPLODED mode
  const fanOffset = exploded ? 1.6 : 0;
  const compressorOffset = exploded ? 0.45 : 0;
  const casingOffset = exploded ? -0.4 : 0;
  const turbineOffset = exploded ? -1.3 : 0;
  const nozzleOffset = exploded ? -2.2 : 0;

  return (
    <group rotation={[0.07, -0.29, -0.13]} scale={1.05}>
      <ambientLight intensity={1.15} />
      <hemisphereLight intensity={1.4} color="#d5e4f3" groundColor="#231e18" />
      <spotLight position={[3, 7, 7]} intensity={85} angle={0.48} penumbra={0.8} decay={2} castShadow color="#ffffff" />
      <spotLight position={[-7, -2, 4]} intensity={50} angle={0.6} penumbra={1} decay={2} color="#b7ccc8" />
      <pointLight position={[1, -2, -5]} intensity={38} decay={2} color="#ff774c" />

      {/* 1. FAN ASSEMBLY & INTAKE LIP */}
      <group
        position={[0, 0, fanOffset]}
        onClick={(e) => { e.stopPropagation(); onSelectPart?.(selectedPart === 'FAN' ? null : 'FAN'); }}
      >
        <mesh position={[0, 0, 1.35]}>
          <torusGeometry args={[1.66, 0.16, 14, 96]} />
          <meshStandardMaterial
            color={selectedPart === 'FAN' ? '#FF5E39' : '#929f9b'}
            metalness={0.96}
            roughness={0.26}
          />
        </mesh>
        <mesh position={[0, 0, 1.42]}>
          <torusGeometry args={[1.42, 0.022, 10, 96]} />
          <meshStandardMaterial color="#e6e8db" metalness={0.8} roughness={0.3} />
        </mesh>
        <BladedStage
          count={23}
          radius={1.22}
          hub={0.27}
          z={1.28}
          rotationSpeed={0.28}
          speed={speed}
          paused={paused}
          reducedMotion={reducedMotion}
          color="#c1c8c1"
          highlighted={selectedPart === 'FAN'}
        />
        <mesh position={[0, 0, 1.53]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <coneGeometry args={[0.36, 0.74, 48]} />
          <meshStandardMaterial
            color={selectedPart === 'FAN' ? '#FFA085' : '#d8dfd6'}
            metalness={0.88}
            roughness={0.2}
          />
        </mesh>
        <HotspotPin
          position={[0, 1.45, 1.35]}
          label="Fan Stage"
          active={selectedPart === 'FAN'}
          onClick={() => onSelectPart?.(selectedPart === 'FAN' ? null : 'FAN')}
        />
      </group>

      {/* 2. COMPRESSOR SECTION (LPC & HPC STAGES) */}
      <group
        position={[0, 0, compressorOffset]}
        onClick={(e) => { e.stopPropagation(); onSelectPart?.(selectedPart === 'COMPRESSOR' ? null : 'COMPRESSOR'); }}
      >
        <BladedStage
          count={17}
          radius={0.83}
          hub={0.25}
          z={0.32}
          rotationSpeed={0.38}
          speed={speed}
          paused={paused}
          reducedMotion={reducedMotion}
          color="#828f89"
          highlighted={selectedPart === 'COMPRESSOR'}
        />
        <BladedStage
          count={21}
          radius={0.71}
          hub={0.25}
          z={-0.49}
          rotationSpeed={0.42}
          speed={speed}
          paused={paused}
          reducedMotion={reducedMotion}
          color="#8b9c97"
          highlighted={selectedPart === 'COMPRESSOR'}
        />
        <HotspotPin
          position={[0, 0.95, -0.1]}
          label="Compressor Stages"
          active={selectedPart === 'COMPRESSOR'}
          onClick={() => onSelectPart?.(selectedPart === 'COMPRESSOR' ? null : 'COMPRESSOR')}
        />
      </group>

      {/* 3. MAIN CASING & STRUCTURAL HOUSING */}
      <group
        position={[0, 0, casingOffset]}
        onClick={(e) => { e.stopPropagation(); onSelectPart?.(selectedPart === 'CASING' ? null : 'CASING'); }}
      >
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[1.63, 1.25, 2.72, 80, 1, true]} />
          <meshStandardMaterial
            color={selectedPart === 'CASING' ? '#FF5E39' : '#39423e'}
            metalness={0.84}
            roughness={0.34}
            transparent
            opacity={xray ? 0.08 : (selectedPart === 'CASING' ? 0.45 : 0.27)}
            side={THREE.DoubleSide}
            depthWrite={!xray}
          />
        </mesh>
        {[0.94, 0.47, -0.1, -0.7, -1.24].map((z, i) => (
          <mesh key={z} position={[0, 0, z]}>
            <torusGeometry args={[1.59 - i * 0.075, 0.037, 10, 80]} />
            <meshStandardMaterial
              color={i === 0 ? '#cdd4c9' : '#70807b'}
              metalness={0.96}
              roughness={0.25}
              transparent
              opacity={xray ? 0.38 : 1}
            />
          </mesh>
        ))}
        {/* Core rotor shaft */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.22, 0.22, 3.22, 30]} />
          <meshStandardMaterial color="#c7d1c8" metalness={1} roughness={0.22} />
        </mesh>
        {/* Exterior struts */}
        {Array.from({ length: 8 }, (_, i) => {
          const a = (i * Math.PI) / 4;
          return (
            <mesh key={i} position={[Math.cos(a) * 1.53, Math.sin(a) * 1.53, -0.05]} rotation={[0, 0, a]}>
              <boxGeometry args={[0.07, 0.09, 2.4]} />
              <meshStandardMaterial
                color="#77827b"
                metalness={0.87}
                roughness={0.32}
                transparent
                opacity={xray ? 0.15 : 0.88}
              />
            </mesh>
          );
        })}
      </group>

      {/* 4. TURBINE SECTION (HPT & LPT) */}
      <group
        position={[0, 0, turbineOffset]}
        onClick={(e) => { e.stopPropagation(); onSelectPart?.(selectedPart === 'TURBINE' ? null : 'TURBINE'); }}
      >
        <mesh position={[0, 0, -1.45]}>
          <torusGeometry args={[1.27, 0.10, 14, 80]} />
          <meshStandardMaterial
            color={selectedPart === 'TURBINE' ? '#FF5E39' : '#bdc7bc'}
            metalness={0.91}
            roughness={0.3}
          />
        </mesh>
        <BladedStage
          count={19}
          radius={0.77}
          hub={0.22}
          z={-1.43}
          rotationSpeed={-0.2}
          speed={speed}
          paused={paused}
          reducedMotion={reducedMotion}
          color="#adb5ac"
          highlighted={selectedPart === 'TURBINE'}
        />
        <HotspotPin
          position={[0, 0.9, -1.45]}
          label="Aft Turbine"
          active={selectedPart === 'TURBINE'}
          onClick={() => onSelectPart?.(selectedPart === 'TURBINE' ? null : 'TURBINE')}
        />
      </group>

      {/* 5. EXHAUST & NOZZLE CONE */}
      <group
        position={[0, 0, nozzleOffset]}
        onClick={(e) => { e.stopPropagation(); onSelectPart?.(selectedPart === 'NOZZLE' ? null : 'NOZZLE'); }}
      >
        <mesh position={[0, 0, -1.65]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[1.25, 0.91, 0.55, 64, 1, true]} />
          <meshStandardMaterial
            color={selectedPart === 'NOZZLE' ? '#FF5E39' : '#535e59'}
            metalness={0.91}
            roughness={0.32}
            side={THREE.DoubleSide}
            transparent
            opacity={xray ? 0.26 : 0.88}
          />
        </mesh>
        <mesh position={[0, 0, -1.97]}>
          <torusGeometry args={[0.92, 0.055, 12, 80]} />
          <meshStandardMaterial color="#cdd3c9" metalness={0.9} roughness={0.27} />
        </mesh>
      </group>

      <ContactShadows position={[0, -2.1, 0]} opacity={0.35} scale={8} blur={2.6} far={4} resolution={256} />
    </group>
  );
}

export default function EngineScene(props: Props) {
  return (
    <Canvas
      shadows
      dpr={[1, 1.7]}
      camera={{ position: [3.65, 2.2, 6.7], fov: 36, near: 0.1, far: 100 }}
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
    >
      <Turbine {...props} />
      <OrbitControls
        enablePan={false}
        minDistance={4.5}
        maxDistance={10}
        minPolarAngle={0.25}
        maxPolarAngle={Math.PI - 0.25}
        enableDamping
        dampingFactor={0.06}
      />
    </Canvas>
  );
}
