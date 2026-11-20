# AeroPulse / flagship UI concept

This is a real starter frontend (not a screenshot): React, Vite, TypeScript, React Three Fiber, original procedural turbofan geometry, scene controls, responsive aerospace editorial UI and interaction prototypes. It is **not** the completed ML product. It does not include NASA raw data, any model checkpoint, or measured predictions. The interactive waveform is deliberately labeled as a visual placeholder.

```bash
npm install
npm run typecheck
npm run build
npm run dev
```

Agent: evolve this visual system into the working product defined in `../../prompts/MASTER_PROMPT.md`; use real API data, do not replace disconnected values with invented ones, and maintain the strong identity. The demo's section anchor navigation and engine controls are functional. Later implement separate route views and deeper 3D interactions. The workbench mode controls affect the shared hero 3D engine; future implementation should put a second persistent 3D viewer directly into the workbench or add an app-wide inspection panel.
