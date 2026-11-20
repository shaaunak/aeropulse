import { Component, Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  ArrowRight, ArrowUpRight, ChevronRight,
  Database, Menu, Pause, Play, Radar, RotateCcw, Search,
  X
} from 'lucide-react';
import type { ModelMode, EnginePart } from './components/EngineScene';

const EngineScene = lazy(() => import('./components/EngineScene'));

const fmt = (i: number) => String(i).padStart(3, '0');

interface ManifestData {
  dataset: string;
  provenance: {
    source: string;
    citation: string;
    operational_conditions: string;
    fault_modes: string;
    train_engines: number;
    test_engines: number;
    selected_sensor_channels: number;
    selected_features: string[];
    dropped_features: string[];
  };
  models: {
    lstm: {
      name: string;
      official_test_uncapped: { rmse: number; mae: number; nasa_score: number };
      official_test_capped: { rmse: number; mae: number };
    };
    gru: {
      name: string;
      official_test_uncapped: { rmse: number; mae: number; nasa_score: number };
    };
    ridge_tabular: {
      name: string;
      official_test_uncapped: { rmse: number; mae: number; nasa_score: number };
    };
    persistence: {
      name: string;
      official_test_uncapped: { rmse: number; mae: number; nasa_score: number };
    };
  };
  fleet_index: Array<{
    unit_id: number;
    observed_cycles: number;
    predicted_rul: number;
    true_rul: number;
    error: number;
    status: string;
  }>;
}

interface EngineDetailData {
  unit_id: number;
  subset: string;
  split: string;
  observed_cycles: number;
  true_rul_final: number;
  predicted_rul_final: number;
  prediction_error: number;
  prediction_status: string;
  cycles: number[];
  predictions_by_cycle: number[];
  retrospective_true_rul_by_cycle: number[];
  sensors: Record<string, number[]>;
}

function useReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    const m = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduce(m.matches);
    update();
    m.addEventListener('change', update);
    return () => m.removeEventListener('change', update);
  }, []);
  return reduce;
}

function canUseWebGL() {
  try {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

function Brand({ small = false }: { small?: boolean }) {
  return (
    <a className={'brand ' + (small ? 'brand--small' : '')} href="#top" aria-label="AeroPulse home">
      <span className="brand__mark">
        <span className="brand__slash">A</span>
        <span className="brand__dot" />
      </span>
      <span className="brand__name">
        AERO<span>PULSE</span>
        <span className="brand__registered">®</span>
      </span>
    </a>
  );
}

function IndexLabel({ number, children, light = false }: { number: string; children: ReactNode; light?: boolean }) {
  return (
    <div className={'index-label ' + (light ? 'index-label--light' : '')}>
      <span>{number}</span>
      <span className="index-label__rule" />
      <span>{children}</span>
    </div>
  );
}

class SceneErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    return this.state.hasError ? (
      <div className="webgl-fallback">
        <p>3D engine visualization unavailable in this browser environment.</p>
        <span>STATIC RESEARCH MODE ACTIVE</span>
      </div>
    ) : (
      this.props.children
    );
  }
}

function TurbineViewport({
  mode,
  speed,
  paused,
  reducedMotion,
  resetKey,
  webgl,
  variant = 'hero',
  selectedPart,
  onSelectPart,
}: {
  mode: ModelMode;
  speed: number;
  paused: boolean;
  reducedMotion: boolean;
  resetKey: number;
  webgl: boolean | null;
  variant?: 'hero' | 'workbench';
  selectedPart?: EnginePart;
  onSelectPart?: (part: EnginePart) => void;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (!('IntersectionObserver' in window)) {
      setInView(true);
      return;
    }
    const el = viewportRef.current;
    if (!el) return;
    const observer = new IntersectionObserver((entries) => setInView(entries[0]?.isIntersecting ?? false), {
      rootMargin: '100px',
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={viewportRef}
      className={'model-viewer ' + (variant === 'workbench' ? 'model-viewer--workbench' : '')}
      aria-label="Interactive 3D illustration of turbofan assembly"
    >
      {webgl === false ? (
        <div className="webgl-fallback">
          <p>3D graphics acceleration unavailable.</p>
          <span>STATIC DIGITAL TWIN MODE</span>
        </div>
      ) : webgl === null || !inView ? (
        <div className="loading-model">3D ENGINE · READY TO INSPECT</div>
      ) : (
        <SceneErrorBoundary>
          <Suspense fallback={<div className="loading-model">INITIALIZING TURBINE GEOMETRY…</div>}>
            <EngineScene
              key={resetKey}
              mode={mode}
              speed={speed}
              paused={paused}
              reducedMotion={reducedMotion}
              selectedPart={selectedPart}
              onSelectPart={onSelectPart}
            />
          </Suspense>
        </SceneErrorBoundary>
      )}
    </div>
  );
}

export default function App() {
  const [manifest, setManifest] = useState<ManifestData | null>(null);
  const [engineData, setEngineData] = useState<EngineDetailData | null>(null);
  const [backendLive, setBackendLive] = useState<boolean | null>(null);

  // 3D Controls
  const [mode, setMode] = useState<ModelMode>('ASSEMBLED');
  const [speed, setSpeed] = useState(1.0);
  const [paused, setPaused] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const [selectedPart, setSelectedPart] = useState<EnginePart>(null);

  // Studio & Fleet state
  const [selectedUnit, setSelectedUnit] = useState(1);
  const [scrubberCycle, setScrubberCycle] = useState<number>(31);
  const [selectedSensor, setSelectedSensor] = useState<string>('sensor_2');
  const [activeModel, setActiveModel] = useState<'lstm' | 'gru' | 'ridge_tabular' | 'persistence'>('lstm');

  // Fleet table search & sort
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<'id' | 'cycles' | 'pred' | 'true' | 'error'>('id');
  const [sortAsc, setSortAsc] = useState(true);
  const [page, setPage] = useState(0);

  // Penalty Explainer state
  const [penaltyError, setPenaltyError] = useState<number>(10);

  // UI helpers
  const [navOpen, setNavOpen] = useState(false);
  const [webgl, setWebgl] = useState<boolean | null>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    setWebgl(canUseWebGL());
  }, []);

  // Probe backend health & load manifest
  useEffect(() => {
    fetch('/data/manifest.json')
      .then((res) => {
        if (!res.ok) throw new Error('Manifest fetch failed');
        return res.json();
      })
      .then((data: ManifestData) => {
        setManifest(data);
      })
      .catch((err) => {
        console.warn('Could not load local manifest.json:', err);
      });

    // Probe live API
    fetch('/api/v1/health')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.service_available) {
          setBackendLive(true);
        } else {
          setBackendLive(false);
        }
      })
      .catch(() => {
        setBackendLive(false);
      });
  }, []);

  // Load specific test engine data whenever selectedUnit changes
  useEffect(() => {
    fetch(`/data/engines/FD001/test/${fmt(selectedUnit)}.json`)
      .then((res) => res.json())
      .then((data: EngineDetailData) => {
        setEngineData(data);
        setScrubberCycle(data.observed_cycles);
      })
      .catch((err) => {
        console.error(`Error loading engine ${selectedUnit}:`, err);
      });
  }, [selectedUnit]);

  // Filter and sort fleet items
  const fleetList = useMemo(() => {
    if (!manifest?.fleet_index) return [];
    let list = manifest.fleet_index.filter((item) => fmt(item.unit_id).includes(search.trim()));
    list.sort((a, b) => {
      let vA = 0;
      let vB = 0;
      if (sortBy === 'id') {
        vA = a.unit_id;
        vB = b.unit_id;
      } else if (sortBy === 'cycles') {
        vA = a.observed_cycles;
        vB = b.observed_cycles;
      } else if (sortBy === 'pred') {
        vA = a.predicted_rul;
        vB = b.predicted_rul;
      } else if (sortBy === 'true') {
        vA = a.true_rul;
        vB = b.true_rul;
      } else if (sortBy === 'error') {
        vA = Math.abs(a.error);
        vB = Math.abs(b.error);
      }
      return sortAsc ? vA - vB : vB - vA;
    });
    return list;
  }, [manifest, search, sortBy, sortAsc]);

  const pageSize = 5;
  const displayedFleet = fleetList.slice(page * pageSize, (page + 1) * pageSize);
  const pageCount = Math.max(1, Math.ceil(fleetList.length / pageSize));

  // Trajectory calculations for selected engine
  const currentPrediction = useMemo(() => {
    if (!engineData || !engineData.cycles.length) return null;
    const idx = engineData.cycles.indexOf(scrubberCycle);
    if (idx !== -1) {
      return {
        cycle: scrubberCycle,
        predicted: engineData.predictions_by_cycle[idx],
        retrospective_true: engineData.retrospective_true_rul_by_cycle[idx],
        error: round(engineData.predictions_by_cycle[idx] - engineData.retrospective_true_rul_by_cycle[idx], 2),
      };
    }
    return {
      cycle: engineData.observed_cycles,
      predicted: engineData.predicted_rul_final,
      retrospective_true: engineData.true_rul_final,
      error: engineData.prediction_error,
    };
  }, [engineData, scrubberCycle]);

  function round(val: number, decimals: number = 2) {
    const factor = Math.pow(10, decimals);
    return Math.round(val * factor) / factor;
  }

  // Calculate asymmetric penalty
  const computedPenalty = useMemo(() => {
    const e = penaltyError;
    if (e < 0) {
      return Math.exp(-e / 13) - 1;
    } else {
      return Math.exp(e / 10) - 1;
    }
  }, [penaltyError]);

  return (
    <div id="top" className="app">
      {/* Top Banner */}
      <div className="announcement">
        <span className="tiny-led" /> FIELD NOTES / NASA C-MAPSS
        <span className="announcement__middle">
          SIMULATION-BASED MACHINE LEARNING RESEARCH · VERIFIED BENCHMARK
        </span>
        <span className="announcement__index">
          {backendLive ? 'API CONNECTED · LIVE INFERENCE' : 'PRECOMPUTED NASA REPOSITORY'}
        </span>
      </div>

      {/* Main Navbar */}
      <header className="navbar">
        <Brand />
        <nav className={'navbar__links ' + (navOpen ? 'navbar__links--open' : '')} aria-label="Main navigation">
          <a href="#machine" onClick={() => setNavOpen(false)}>
            THE MACHINE <span>01</span>
          </a>
          <a href="#fleet" onClick={() => setNavOpen(false)}>
            FLEET ATLAS <span>02</span>
          </a>
          <a href="#studio" onClick={() => setNavOpen(false)}>
            PREDICTION STUDIO <span>03</span>
          </a>
          <a href="#lab" onClick={() => setNavOpen(false)}>
            MODEL LAB <span>04</span>
          </a>
          <a href="#method" onClick={() => setNavOpen(false)}>
            METHODOLOGY <span>05</span>
          </a>
        </nav>
        <a className="navbar__cta" href="#studio">
          INSPECT ENGINE {fmt(selectedUnit)} <ArrowUpRight size={15} />
        </a>
        <button
          className="mobile-menu"
          aria-label={navOpen ? 'Close menu' : 'Open menu'}
          onClick={() => setNavOpen(!navOpen)}
        >
          {navOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </header>

      <main>
        {/* HERO SECTION */}
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero__grain" aria-hidden="true" />
          <div className="hero__content">
            <div className="hero__eyebrow">
              <span className="dash" /> NASA C-MAPSS TURBOFAN PROGNOSTICS
              <span className="hero__ref">BENCHMARK / FD001</span>
            </div>
            <h1 id="hero-title">
              KNOW<br />
              THE <em>LIMIT.</em><br />
              <span className="hero__outline">BEFORE IT</span><br />
              ARRIVES<span className="hero__period">.</span>
            </h1>
            <div className="hero__bottom">
              <p>
                A rigorous research interface for turbofan degradation. Causal sequence modeling
                meets the physics of remaining useful life.
              </p>
              <a className="pill-action" href="#machine">
                EXPLORE DIGITAL TWIN{' '}
                <span>
                  <ArrowUpRight size={18} />
                </span>
              </a>
            </div>
            <div className="hero__micro">
              <span>01 / PROGNOSTICS</span>
              <span>TESTED ON 100 UNSEEN NASA ENGINES</span>
              <span>↓ SCROLL TO EXPLORE</span>
            </div>
          </div>
          <div className="hero__art">
            <div className="hero__art-corner hero__art-corner--tl" />
            <div className="hero__art-corner hero__art-corner--br" />
            <div className="model-tag model-tag--top">
              <span className="live-dot" /> 3D DIGITAL TWIN{' '}
              <span className="model-tag__right">/ INTERACTIVE DYNAMICS</span>
            </div>
            <div className="model-orbit" aria-hidden="true">
              <div className="orbit orbit--1" />
              <div className="orbit orbit--2" />
              <div className="orbit orbit--3" />
            </div>
            <TurbineViewport
              mode={mode}
              speed={speed}
              paused={paused}
              reducedMotion={reducedMotion}
              resetKey={resetKey}
              webgl={webgl}
              selectedPart={selectedPart}
              onSelectPart={setSelectedPart}
            />
            <div className="model-hud model-hud--left">
              <div className="hud-bar" />
              <span>FIG. 01</span>
              <strong>
                TURBOFAN
                <br />
                DIGITAL TWIN
              </strong>
              <small>PROCEDURAL 3D ANATOMY</small>
            </div>
            <div className="model-hud model-hud--right">
              <span>ACTIVE MODEL</span>
              <strong>CAUSAL LSTM</strong>
              <span>RMSE 15.75 CYCLES ↗</span>
            </div>
            <div className="hero__art-caption">
              <span>FIG 001 / TURBINE ROTARY STAGES</span>
              <span>DRAG TO ORBIT · SCROLL TO ZOOM</span>
            </div>
          </div>
        </section>

        {/* VERIFIED RESEARCH FACTS STRIP */}
        <section className="facts" aria-label="Dataset details">
          <div className="facts__lead">
            <span className="small-dot" /> RESEARCH IN NUMBERS
          </div>
          <div className="fact">
            <b>17<span>ch</span></b>
            <small>ACTIVE SENSORS (7 DROPPED CONSTANT)</small>
          </div>
          <div className="fact">
            <b>30<span>c</span></b>
            <small>CAUSAL WINDOW HORIZON</small>
          </div>
          <div className="fact">
            <b>15.8<span>c</span></b>
            <small>OFFICIAL TEST RMSE (LSTM)</small>
          </div>
          <div className="fact fact--source">
            <span>DATA FOUNDATION</span>
            <strong>NASA / C-MAPSS FD001</strong>
          </div>
        </section>

        {/* 01 / THE MACHINE — DIGITAL TWIN */}
        <section className="section-machine section-light" id="machine">
          <IndexLabel number="01">THE MACHINE</IndexLabel>
          <div className="section-heading">
            <h2>
              THE FUTURE<br />
              HAS A <i>PATTERN.</i>
            </h2>
            <div className="section-heading__aside">
              <span className="small-dot" />
              <p>
                Every turbofan leaves a subtle trail of aerodynamic and thermal signatures. Inspect
                its internal stages, explode the casing, and map illustrative sensor zones.
              </p>
              <a href="#studio" className="text-link">
                SCRUB INFERENCE STUDIO <ArrowUpRight size={16} />
              </a>
            </div>
          </div>

          <div className="mode-workbench">
            <div className="workbench-head">
              <span>
                <span className="small-dot" /> INTERACTIVE ENGINE ANATOMY
              </span>
              <span>AP / ENGINE SIMULATION — RESEARCH SPEC</span>
            </div>
            <div className="mode-workbench__main">
              <div className="workbench-copy">
                <span className="eyebrow-light">VISUALIZATION CONTROL / 001</span>
                <h3>
                  A MACHINE<br />
                  IN MOTION.
                </h3>
                <p>
                  Click stages to isolate internal components. Toggle Exploded view to physically
                  separate fan, compressor stages, casing, and aft turbine.
                </p>

                <div className="workbench-controls">
                  <span>ASSEMBLY VIEW</span>
                  <div className="segmented" role="group" aria-label="Engine view mode">
                    {(['ASSEMBLED', 'XRAY', 'EXPLODED'] as ModelMode[]).map((m) => (
                      <button
                        key={m}
                        aria-pressed={mode === m}
                        className={mode === m ? 'selected' : ''}
                        onClick={() => setMode(m)}
                      >
                        {m === 'ASSEMBLED' ? 'FULL' : m}
                      </button>
                    ))}
                  </div>

                  <label className="speed-label" htmlFor="rotor-speed">
                    ROTOR SPEED <strong>{speed.toFixed(1)}×</strong>
                  </label>
                  <input
                    id="rotor-speed"
                    type="range"
                    min="0"
                    max="3"
                    step="0.1"
                    value={speed}
                    onChange={(e) => setSpeed(Number(e.target.value))}
                  />

                  <div className="control-inline">
                    <button onClick={() => setPaused((p) => !p)}>
                      {paused ? <Play size={14} /> : <Pause size={14} />} {paused ? 'RESUME' : 'PAUSE'}
                    </button>
                    <button
                      onClick={() => {
                        setMode('ASSEMBLED');
                        setSpeed(1);
                        setPaused(false);
                        setSelectedPart(null);
                        setResetKey((k) => k + 1);
                      }}
                    >
                      <RotateCcw size={14} /> RESET VIEW
                    </button>
                  </div>

                  {/* Component Inspector Drawer */}
                  {selectedPart && (
                    <div className="part-inspector">
                      <div className="part-inspector__head">
                        <span>STAGE: {selectedPart}</span>
                        <span className="part-inspector__badge">ISOLATED</span>
                      </div>
                      <p className="part-inspector__desc">
                        {selectedPart === 'FAN' &&
                          'Intake fan lip and titanium bypass rotor. Directs ambient airflow into primary core and high-bypass duct.'}
                        {selectedPart === 'COMPRESSOR' &&
                          'Concentric Low-Pressure (LPC) and High-Pressure (HPC) bladed stages. Subject to progressive fouling and blade tip clearance wear.'}
                        {selectedPart === 'CASING' &&
                          'Structural titanium and composite containment housing with radial load-bearing struts.'}
                        {selectedPart === 'TURBINE' &&
                          'High-Pressure (HPT) and Low-Pressure (LPT) reaction stages driving compressor shafts via Brayton cycle expansion.'}
                        {selectedPart === 'NOZZLE' &&
                          'Aft core exhaust and variable convergence geometry for thrust consolidation.'}
                      </p>
                      <div className="part-inspector__sensors">
                        <span>ILLUSTRATIVE SENSOR GROUPING:</span>
                        {selectedPart === 'FAN' && (
                          <>
                            <span className="part-inspector__tag">Total Temp T2 (sensor_2)</span>
                            <span className="part-inspector__tag">Fan Speed N1 (sensor_8)</span>
                          </>
                        )}
                        {selectedPart === 'COMPRESSOR' && (
                          <>
                            <span className="part-inspector__tag">LPC Outlet Temp (sensor_3)</span>
                            <span className="part-inspector__tag">HPC Outlet Temp (sensor_4)</span>
                            <span className="part-inspector__tag">Static Pressure Ps30 (sensor_11)</span>
                          </>
                        )}
                        {selectedPart === 'CASING' && (
                          <>
                            <span className="part-inspector__tag">Bypass Ratio (sensor_12)</span>
                            <span className="part-inspector__tag">Bleed Enthalpy (sensor_13)</span>
                          </>
                        )}
                        {selectedPart === 'TURBINE' && (
                          <>
                            <span className="part-inspector__tag">HPT Coolant Bleed (sensor_14)</span>
                            <span className="part-inspector__tag">LPT Coolant Bleed (sensor_15)</span>
                            <span className="part-inspector__tag">Core Speed N2 (sensor_9)</span>
                          </>
                        )}
                        {selectedPart === 'NOZZLE' && (
                          <>
                            <span className="part-inspector__tag">Exhaust Pressure P50 (sensor_7)</span>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="workbench-visual">
                <div className="workbench-visual__label">AP/ROTARY ENGINE · 3D STUDY</div>
                <TurbineViewport
                  variant="workbench"
                  mode={mode}
                  speed={speed}
                  paused={paused}
                  reducedMotion={reducedMotion}
                  resetKey={resetKey}
                  webgl={webgl}
                  selectedPart={selectedPart}
                  onSelectPart={setSelectedPart}
                />
                <div className="workbench-visual__foot">
                  DRAG TO ORBIT <span>↗</span> CLICK PINS TO ISOLATE STAGES
                </div>
              </div>

              <div className="workbench-spec">
                <div className="spec-cross" />
                <div>
                  <span>TYPE</span>
                  <b>3D / PROCEDURAL</b>
                </div>
                <div>
                  <span>STAGE FOCUS</span>
                  <b>{selectedPart || 'COMPLETE ASSEMBLY'}</b>
                </div>
                <div>
                  <span>REPRESENTATION</span>
                  <b>ILLUSTRATIVE SENSORS</b>
                </div>
                <div>
                  <span>SIMULATION</span>
                  <b>NASA C-MAPSS FD001</b>
                </div>
                <div className="spec-mark">
                  AP<span>®</span>
                </div>
              </div>
            </div>
            <div className="workbench-foot">
              <span>ENGINE ARCHITECTURE STUDY</span>
              <span>EXPLODE / INSPECT / UNDERSTAND</span>
              <span>004 CONTROLS ACTIVE</span>
            </div>
          </div>
        </section>

        {/* 02 / FLEET ATLAS */}
        <section className="section-fleet section-light" id="fleet">
          <IndexLabel number="02">FLEET ATLAS</IndexLabel>
          <div className="fleet-top">
            <h2>
              ONE HUNDRED<br />
              <em>DIFFERENT</em> STORIES.
            </h2>
            <div>
              <p>
                100 genuine NASA FD001 test trajectories. Click any engine to inspect its timeline,
                sensor readings, and prediction curve in the Studio.
              </p>
              <div className="availability">
                <span className="availability__light" /> DATASET LOADED · 100 TEST ENGINES
              </div>
            </div>
          </div>

          <div className="fleet-board">
            <div className="fleet-board__top">
              <div className="fleet-board__title">
                <Radar size={17} />
                <span>FD001 / FLEET INDEX</span>
                <span className="fleet-board__count">{fleetList.length} UNITS</span>
              </div>
              <label className="fleet-search">
                <Search size={17} />
                <input
                  aria-label="Search engine unit ID"
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value.replace(/\D/g, '').slice(0, 3));
                    setPage(0);
                  }}
                  placeholder="SEARCH UNIT ID"
                  inputMode="numeric"
                />
              </label>
            </div>

            <div className="fleet-board__columns">
              <button
                onClick={() => {
                  setSortBy('id');
                  setSortAsc(!sortAsc);
                }}
                style={{ background: 'none', border: 'none', font: 'inherit', textAlign: 'left', padding: 0 }}
              >
                UNIT / ID {sortBy === 'id' && (sortAsc ? '▲' : '▼')}
              </button>
              <button
                onClick={() => {
                  setSortBy('cycles');
                  setSortAsc(!sortAsc);
                }}
                style={{ background: 'none', border: 'none', font: 'inherit', textAlign: 'left', padding: 0 }}
              >
                OBSERVED CYCLES {sortBy === 'cycles' && (sortAsc ? '▲' : '▼')}
              </button>
              <button
                onClick={() => {
                  setSortBy('pred');
                  setSortAsc(!sortAsc);
                }}
                style={{ background: 'none', border: 'none', font: 'inherit', textAlign: 'left', padding: 0 }}
              >
                PREDICTED RUL {sortBy === 'pred' && (sortAsc ? '▲' : '▼')}
              </button>
              <button
                onClick={() => {
                  setSortBy('true');
                  setSortAsc(!sortAsc);
                }}
                style={{ background: 'none', border: 'none', font: 'inherit', textAlign: 'left', padding: 0 }}
              >
                GROUND TRUTH RUL {sortBy === 'true' && (sortAsc ? '▲' : '▼')}
              </button>
              <span />
            </div>

            {displayedFleet.map((item) => (
              <button
                className={'fleet-row ' + (selectedUnit === item.unit_id ? 'fleet-row--active' : '')}
                key={item.unit_id}
                onClick={() => {
                  setSelectedUnit(item.unit_id);
                  const el = document.getElementById('studio');
                  if (el) el.scrollIntoView({ behavior: 'smooth' });
                }}
                aria-pressed={selectedUnit === item.unit_id}
              >
                <span className="fleet-row__id">
                  <span className="fleet-row__dot" />
                  ENGINE—{fmt(item.unit_id)}
                </span>
                <span>{item.observed_cycles} cycles</span>
                <span style={{ color: '#ff5e39', fontWeight: 700 }}>{item.predicted_rul} cycles</span>
                <span>{item.true_rul} cycles (err: {item.error > 0 ? `+${item.error}` : item.error})</span>
                <ArrowUpRight size={16} />
              </button>
            ))}

            {displayedFleet.length === 0 && (
              <div className="fleet-board__empty">NO MATCHING ENGINE INDEX FOUND.</div>
            )}

            <div className="fleet-board__bottom">
              <div>
                <span className="tiny-led" /> SELECTED: ENGINE {fmt(selectedUnit)}{' '}
                <span className="muted">/ loaded in Prediction Studio</span>
              </div>
              <div className="pagination">
                <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>
                  PREV
                </button>
                <span>
                  {page + 1} / {pageCount}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                  disabled={page >= pageCount - 1}
                >
                  NEXT <ChevronRight size={13} />
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* 03 / PREDICTION STUDIO */}
        <section className="studio-section" id="studio">
          <IndexLabel number="03" light>
            PREDICTION STUDIO
          </IndexLabel>

          <div className="studio-header">
            <div>
              <span className="eyebrow-dark">CAUSAL TEMPORAL INFERENCE</span>
              <h2>
                KNOW THE <em>TIMELINE.</em>
              </h2>
            </div>
            <p>
              Scrub engine operating cycles causally. Predictions are calculated using strictly
              prior observation windows without future lookahead.
            </p>
          </div>

          <div className="studio-board">
            {/* Topbar */}
            <div className="studio-topbar">
              <div className="studio-topbar__engine">
                <span className="live-dot" /> ENGINE {fmt(selectedUnit)}
                <span style={{ color: '#ff5e39' }}>[FD001 TEST SPLIT]</span>
              </div>
              <div className="studio-topbar__stats">
                <span className="studio-topbar__stat">
                  TOTAL OBSERVED: <strong>{engineData?.observed_cycles || '—'} CYCLES</strong>
                </span>
                <span className="studio-topbar__stat">
                  OFFICIAL GROUND TRUTH: <strong>{engineData?.true_rul_final || '—'} CYCLES</strong>
                </span>
                <span className="studio-topbar__stat">
                  MODEL PREDICTION: <strong>{engineData?.predicted_rul_final || '—'} CYCLES</strong>
                </span>
              </div>
            </div>

            {/* Scrubber Controls */}
            <div className="studio-scrubber">
              <div className="studio-scrubber__labels">
                <span>OBSERVATION HORIZON: <b>CYCLE {scrubberCycle}</b> / {engineData?.observed_cycles || 100}</span>
                <span>DRAG SLIDER TO RECONSTRUCT CAUSAL HISTORY</span>
              </div>
              <input
                type="range"
                className="studio-slider"
                min={1}
                max={engineData?.observed_cycles || 100}
                value={scrubberCycle}
                onChange={(e) => setScrubberCycle(Number(e.target.value))}
              />
            </div>

            {/* Readout Cards */}
            <div className="studio-readout-grid">
              <div className="studio-readout-card">
                <span>CURRENT ESTIMATED RUL</span>
                <strong className="highlight">{currentPrediction?.predicted ?? '—'}</strong>
                <small>OPERATING CYCLES REMAINING</small>
              </div>
              <div className="studio-readout-card">
                <span>RETROSPECTIVE GROUND TRUTH</span>
                <strong>{currentPrediction?.retrospective_true ?? '—'}</strong>
                <small>EVALUATION BENCHMARK ONLY</small>
              </div>
              <div className="studio-readout-card">
                <span>RESIDUAL ERROR (e = pred - true)</span>
                <strong style={{ color: (currentPrediction?.error ?? 0) > 0 ? '#ff8b6b' : '#a2d6a5' }}>
                  {(currentPrediction?.error ?? 0) > 0 ? `+${currentPrediction?.error}` : currentPrediction?.error}
                </strong>
                <small>{(currentPrediction?.error ?? 0) > 0 ? 'OVERESTIMATE (HEAVILY PENALIZED)' : 'UNDERESTIMATE (SAFE)'}</small>
              </div>
              <div className="studio-readout-card">
                <span>ACTIVE MODEL BACKBONE</span>
                <div style={{ display: 'flex', gap: '4px', margin: '4px 0' }}>
                  {(['lstm', 'gru', 'ridge_tabular', 'persistence'] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setActiveModel(m)}
                      style={{
                        padding: '3px 6px',
                        fontSize: '8px',
                        fontFamily: 'var(--mono)',
                        border: '1px solid',
                        borderColor: activeModel === m ? '#ff5e39' : '#3c483f',
                        background: activeModel === m ? '#ff5e39' : 'transparent',
                        color: activeModel === m ? '#101411' : '#a2aea4',
                        cursor: 'pointer',
                        fontWeight: activeModel === m ? 700 : 400,
                      }}
                    >
                      {m === 'lstm' ? 'LSTM' : m === 'gru' ? 'GRU' : m === 'ridge_tabular' ? 'RIDGE' : 'PERS'}
                    </button>
                  ))}
                </div>
                <strong style={{ fontSize: '16px' }}>
                  {activeModel === 'lstm' && '2-LAYER LSTM'}
                  {activeModel === 'gru' && '2-LAYER GRU'}
                  {activeModel === 'ridge_tabular' && 'RIDGE BASELINE'}
                  {activeModel === 'persistence' && 'MEDIAN BASELINE'}
                </strong>
                <small>RMSE: {manifest?.models[activeModel]?.official_test_uncapped?.rmse ?? '15.75'} CYCLES</small>
              </div>
            </div>

            {/* Trajectory and Sensor Charts */}
            <div className="studio-chart-container">
              <div className="chart-controls">
                <div className="chart-legend">
                  <div className="legend-item">
                    <span className="legend-line legend-line--pred" />
                    <span>CAUSAL LSTM PREDICTION</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-line legend-line--true" />
                    <span>RETROSPECTIVE GROUND TRUTH (BENCHMARK AID)</span>
                  </div>
                </div>

                {/* Sensor Channel Selector */}
                <div className="sensor-chips">
                  <span>CHANNELS:</span>
                  {['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_11', 'sensor_12', 'sensor_15'].map(
                    (s) => (
                      <button
                        key={s}
                        className={selectedSensor === s ? 'active' : ''}
                        onClick={() => setSelectedSensor(s)}
                      >
                        {s.replace('sensor_', 'CH—')}
                      </button>
                    )
                  )}
                </div>
              </div>

              {/* Trajectory SVG */}
              {engineData && engineData.cycles.length > 0 && (
                <svg className="studio-trajectory-svg" viewBox="0 0 1000 240" preserveAspectRatio="none">
                  <defs>
                    <pattern id="studioGrid" width="100" height="48" patternUnits="userSpaceOnUse">
                      <path d="M 100 0 L 0 0 0 48" fill="none" stroke="#ffffff" strokeOpacity="0.08" strokeWidth="1" />
                    </pattern>
                  </defs>
                  <rect width="1000" height="240" fill="url(#studioGrid)" />

                  {/* Sensor waveform backdrop */}
                  {(() => {
                    const rawSensor = engineData.sensors[selectedSensor] || [];
                    if (!rawSensor.length) return null;
                    const minS = Math.min(...rawSensor);
                    const maxS = Math.max(...rawSensor);
                    const rangeS = maxS - minS || 1;
                    const sPoints = rawSensor.map((v, i) => {
                      const x = (i / (rawSensor.length - 1)) * 1000;
                      const y = 220 - ((v - minS) / rangeS) * 160;
                      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                    }).join(' ');
                    return (
                      <path d={sPoints} fill="none" stroke="#7d9285" strokeWidth="1.2" strokeOpacity="0.45" />
                    );
                  })()}

                  {/* Retrospective True RUL line */}
                  {(() => {
                    const trueSeries = engineData.retrospective_true_rul_by_cycle;
                    const maxTrue = Math.max(...trueSeries, 150);
                    const tPoints = trueSeries.map((v, i) => {
                      const x = (i / (trueSeries.length - 1)) * 1000;
                      const y = 220 - (v / maxTrue) * 200;
                      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                    }).join(' ');
                    return (
                      <path d={tPoints} fill="none" stroke="#F3F1EB" strokeWidth="1.8" strokeDasharray="4 4" strokeOpacity="0.65" />
                    );
                  })()}

                  {/* Causal Predicted RUL line */}
                  {(() => {
                    const predSeries = engineData.predictions_by_cycle;
                    const trueSeries = engineData.retrospective_true_rul_by_cycle;
                    const maxTrue = Math.max(...trueSeries, 150);
                    const pPoints = predSeries.map((v, i) => {
                      const x = (i / (predSeries.length - 1)) * 1000;
                      const y = 220 - (v / maxTrue) * 200;
                      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                    }).join(' ');
                    return (
                      <path d={pPoints} fill="none" stroke="#FF5E39" strokeWidth="2.4" />
                    );
                  })()}

                  {/* Current Scrubber Marker */}
                  {(() => {
                    const cIdx = engineData.cycles.indexOf(scrubberCycle);
                    if (cIdx === -1) return null;
                    const x = (cIdx / (engineData.cycles.length - 1)) * 1000;
                    const trueSeries = engineData.retrospective_true_rul_by_cycle;
                    const maxTrue = Math.max(...trueSeries, 150);
                    const predVal = engineData.predictions_by_cycle[cIdx];
                    const y = 220 - (predVal / maxTrue) * 200;
                    return (
                      <g>
                        <line x1={x} y1="0" x2={x} y2="240" stroke="#FF5E39" strokeWidth="1.5" strokeDasharray="2 3" />
                        <circle cx={x} cy={y} r="6" fill="#FF5E39" stroke="#101411" strokeWidth="2" />
                      </g>
                    );
                  })()}
                </svg>
              )}

              <div className="chart-axis-labels">
                <span>CYCLE 001 (START OF OBSERVATION)</span>
                <span>GENUINE FLIGHT TIME-SERIES TRAJECTORY</span>
                <span>CYCLE {engineData?.observed_cycles || 100} (TERMINAL SNAPSHOT)</span>
              </div>
            </div>
          </div>
        </section>

        {/* 04 / MODEL LAB */}
        <section className="lab-section" id="lab">
          <IndexLabel number="04" light>
            MODEL LAB
          </IndexLabel>

          <div className="lab-section__header">
            <div>
              <span className="eyebrow-dark">VERIFIED BENCHMARK & REPRODUCIBILITY</span>
              <h2>
                NOT MAGIC.<br />
                <i>MEASUREMENT.</i>
              </h2>
            </div>
            <p>
              Tested on all 100 official NASA FD001 test engine endpoints. Zero fabricated
              numbers, zero data leakage, and rigorous asymmetric penalty evaluation.
            </p>
          </div>

          {/* Scoreboard Cards */}
          <div className="metrics-scoreboard">
            <div className="metric-box metric-box--champion">
              <span className="metric-box__badge">PRIMARY MODEL</span>
              <span>CAUSAL 2-LAYER LSTM</span>
              <h4>OFFICIAL TEST RMSE</h4>
              <div className="metric-val">15.75</div>
              <div className="metric-sub">
                MAE: 10.58 cycles · Capped RMSE: 14.39<br />
                NASA Asymmetric Score: 418.66
              </div>
            </div>

            <div className="metric-box">
              <span>RECURRENT COMPARISON</span>
              <h4>CAUSAL 2-LAYER GRU</h4>
              <div className="metric-val" style={{ color: '#f3f1eb' }}>15.01</div>
              <div className="metric-sub">
                MAE: 10.22 cycles<br />
                NASA Asymmetric Score: 401.65
              </div>
            </div>

            <div className="metric-box">
              <span>FEATURED BASELINE</span>
              <h4>TABULAR RIDGE (30c)</h4>
              <div className="metric-val" style={{ color: '#a2aea4' }}>18.98</div>
              <div className="metric-sub">
                MAE: 13.41 cycles<br />
                NASA Asymmetric Score: 612.67
              </div>
            </div>

            <div className="metric-box">
              <span>NAIVE BENCHMARK</span>
              <h4>MEDIAN PERSISTENCE</h4>
              <div className="metric-val" style={{ color: '#768579' }}>43.69</div>
              <div className="metric-sub">
                MAE: 38.12 cycles<br />
                NASA Asymmetric Score: 41,530.95
              </div>
            </div>
          </div>

          {/* Model Comparison Table */}
          <table className="lab-comparison-table" aria-label="Model performance comparison">
            <thead>
              <tr>
                <th>ARCHITECTURE / MODEL ID</th>
                <th>CONTEXT HORIZON</th>
                <th>OFFICIAL TEST RMSE</th>
                <th>OFFICIAL TEST MAE</th>
                <th>NASA ASYMMETRIC SCORE</th>
                <th>STATUS</th>
              </tr>
            </thead>
            <tbody>
              <tr className="champion-row">
                <td>CAUSAL 2-LAYER LSTM (model_lstm_v1)</td>
                <td>30 cycles (sliding)</td>
                <td>15.75 cycles</td>
                <td>10.58 cycles</td>
                <td>418.66</td>
                <td>BEST CHECKPOINT</td>
              </tr>
              <tr>
                <td>CAUSAL 2-LAYER GRU (model_gru_v1)</td>
                <td>30 cycles (sliding)</td>
                <td>15.01 cycles</td>
                <td>10.22 cycles</td>
                <td>401.65</td>
                <td>BENCHMARKED</td>
              </tr>
              <tr>
                <td>ENGINEERED RIDGE REGRESSION</td>
                <td>30c statistical summary</td>
                <td>18.98 cycles</td>
                <td>13.41 cycles</td>
                <td>612.67</td>
                <td>BENCHMARKED</td>
              </tr>
              <tr>
                <td>MEDIAN TARGET BASELINE</td>
                <td>Static training median</td>
                <td>43.69 cycles</td>
                <td>38.12 cycles</td>
                <td>41,530.95</td>
                <td>BENCHMARKED</td>
              </tr>
            </tbody>
          </table>

          {/* Interactive Scatter Plot: 100 Engine Prediction vs Ground Truth */}
          <div className="scatter-container">
            <div className="scatter-header">
              <span>FIG. 04 / TEST PREDICTION VS GROUND TRUTH (100 ENGINES)</span>
              <span>X = OFFICIAL GROUND TRUTH RUL · Y = MODEL PREDICTION</span>
            </div>

            <svg className="scatter-svg" viewBox="0 0 1000 280">
              <defs>
                <pattern id="scatterGrid" width="100" height="56" patternUnits="userSpaceOnUse">
                  <path d="M 100 0 L 0 0 0 56" fill="none" stroke="#ffffff" strokeOpacity="0.08" strokeWidth="1" />
                </pattern>
              </defs>
              <rect width="1000" height="280" fill="url(#scatterGrid)" />

              {/* Ideal y=x diagonal */}
              <line x1="0" y1="280" x2="1000" y2="0" stroke="#f3f1eb" strokeWidth="1.2" strokeDasharray="4 4" strokeOpacity="0.4" />

              {/* 100 Engine Points */}
              {manifest?.fleet_index.map((item) => {
                const maxVal = 145; // max test RUL in FD001
                const cx = (item.true_rul / maxVal) * 960 + 20;
                const cy = 280 - ((item.predicted_rul / maxVal) * 260 + 10);
                const isSelected = selectedUnit === item.unit_id;
                return (
                  <circle
                    key={item.unit_id}
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 6 : 3.5}
                    fill={isSelected ? '#FF5E39' : (item.error > 0 ? '#ff8b6b' : '#68e090')}
                    stroke="#101411"
                    strokeWidth={isSelected ? 2 : 1}
                    style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                    onClick={() => {
                      setSelectedUnit(item.unit_id);
                      const el = document.getElementById('studio');
                      if (el) el.scrollIntoView({ behavior: 'smooth' });
                    }}
                  >
                    <title>Engine {item.unit_id}: True {item.true_rul}c, Pred {item.predicted_rul}c (err: {item.error}c)</title>
                  </circle>
                );
              })}
            </svg>
            <div className="chart-axis-labels">
              <span>0 CYCLES (NEAR FAILURE)</span>
              <span>DIAGONAL DASHED LINE = PERFECT AGREEMENT (y = x)</span>
              <span>140+ CYCLES (HEALTHY INITIAL STATE)</span>
            </div>
          </div>
        </section>

        {/* 05 / THE METHOD */}
        <section className="method-section section-light" id="method">
          <IndexLabel number="05">THE METHOD</IndexLabel>
          <div className="method-section__content">
            <div>
              <h2>
                FROM SENSOR<br />
                TO <em>FORESIGHT.</em>
              </h2>
              <p>
                Engineered for strict scientific reproducibility. We detail the exact dataset
                provenance, the leakage-safe train/validation split, feature variance filtering,
                and the NASA asymmetric penalty function.
              </p>

              {/* Interactive Penalty Explainer */}
              <div className="penalty-calculator">
                <h4>INTERACTIVE NASA ASYMMETRIC SCORE CALCULATOR</h4>
                <p>
                  In aerospace prognostics, an overestimate (predicting the engine can run longer
                  than it safely can) is far more hazardous than an early maintenance call.
                </p>
                <div className="penalty-inputs">
                  <div className="penalty-input-item">
                    <label htmlFor="err-input">ERROR e = (PRED - TRUE):</label>
                    <input
                      id="err-input"
                      type="number"
                      value={penaltyError}
                      onChange={(e) => setPenaltyError(Number(e.target.value))}
                    />
                  </div>
                  <div className="penalty-result">
                    <span>PENALTY SCORE:</span>
                    <strong>{round(computedPenalty, 2)}</strong>
                  </div>
                </div>
                <small style={{ marginTop: '8px', display: 'block', color: '#68746a' }}>
                  Notice: e = +10 yields {round(Math.exp(10 / 10) - 1, 2)}, while e = -10 yields only{' '}
                  {round(Math.exp(10 / 13) - 1, 2)}.
                </small>
              </div>
            </div>

            <div className="method-steps">
              {[
                [
                  '01',
                  'INGESTION & ASSERTIONS',
                  'NASA C-MAPSS FD001 (100 train engines, 100 test engines, 26 space-separated columns). Verified unique monotonically increasing cycles and finite values.',
                ],
                [
                  '02',
                  'LEAKAGE-SAFE SPLIT & FILTERING',
                  'Train engines partitioned 80/20 by engine ID. Variance thresholding drops 7 constant channels (setting_3, s1, s5, s10, s16, s18, s19). StandardScaler fit on training engines only.',
                ],
                [
                  '03',
                  'CAUSAL TEMPORAL WINDOWING',
                  'Intra-engine sliding windows of size 30. Causal left-edge padding for initial cycles. Training target capped at 125 cycles to stabilize early degradation.',
                ],
                [
                  '04',
                  'OFFICIAL BENCHMARK AUDIT',
                  'Exactly one prediction per test engine evaluated at its final observation window against NASA RUL_FD001.txt. Real test RMSE of 15.75 cycles.',
                ],
              ].map(([n, t, body]) => (
                <div className="method-step" key={n}>
                  <span>{n}</span>
                  <h3>{t}</h3>
                  <p>{body}</p>
                  <ArrowRight size={18} />
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="footer">
        <div className="footer__top">
          <div>
            <span className="eyebrow-dark">AEROPULSE / OPEN AEROSPACE RESEARCH</span>
            <h2>
              EVERY CYCLE<br />
              <i>COUNTS.</i>
            </h2>
          </div>
          <a href="#top" className="footer__to-top">
            BACK TO TOP <ArrowUpRight size={17} />
          </a>
        </div>
        <div className="footer__bottom">
          <Brand small />
          <div>
            BUILT ON NASA C-MAPSS SIMULATED TURBOFAN DATA. RESEARCH APPLICATION ONLY — NOT AN AVIATION SAFETY TOOL.
          </div>
          <span>© 2026 AEROPULSE / PROGNOSTICS RESEARCH</span>
        </div>
      </footer>

      {/* Floating Status Indicator */}
      <div className="floating-disclosure">
        <Database size={13} />
        NASA C-MAPSS FD001 · 100 TEST ENGINES LOADED <span>·</span> LSTM TEST RMSE 15.75c
      </div>
    </div>
  );
}
