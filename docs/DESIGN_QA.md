# AeroPulse Design QA & Visual Identity Audit

## 1. Design Overview
- **Aesthetic Thesis**: Contemporary Swiss editorial meets aerospace instrumentation craftsmanship.
- **Color Palette**:
  - Ink Graphite: `#101411` / `#171C1A`
  - Drafting Paper: `#F3F1EB` / `#E8E5DC`
  - Signal Vermilion: `#FF5E39`
  - Technical Acid Accent: `#DAEFAB` (used sparingly for subtle LEDs)
- **Typography**: Narrow editorial grotesque display paired with `IBM Plex Mono` tabular numerals and clean sans-serif body.

---

## 2. Viewport Inspection & Audit Ledger

### A. Desktop (1440 × 900 Viewport)
- **Screenshot Evidence**: `docs/screenshots/audit_desktop_hero_1440x900.png`, `docs/screenshots/desktop_webgl_rendered.png`
- **Hero & 3D Twin**: Oversized typography scales cleanly without clipping; 3D procedural turbofan renders with brushed metal reflections and real-time orbit controls.
- **Fleet Atlas**: 5-column ledger layout with sorting buttons and search filter.
- **Prediction Studio**: 4-card metric readout grid and dual-line SVG trajectory chart with live cycle marker.
- **Model Lab**: 4-column scoreboard, comparison table, and 100-engine residual scatter plot.
- **Methodology**: Two-column layout with interactive asymmetric score calculator.

### B. Mobile (390 × 844 iPhone Viewport)
- **Screenshot Evidence**: `docs/screenshots/audit_mobile_hero_390x844.png`, `docs/screenshots/audit_mobile_studio_390x844.png`
- **Navigation**: Clean header collapse into accessible hamburger menu toggle.
- **Layout Flow**: Two-column desktop grids gracefully collapse to single-column card stacks.
- **Overflow Check**: Zero horizontal scroll or container blowouts (`overflow-x: hidden`).
- **Touch Targets**: All buttons, slider thumbs, and table rows exceed minimum 44px touch requirements.

---

## 3. 3D WebGL & Low-Power Fallback
- **Lazy Rendering**: Three.js canvas dynamically mounts via `IntersectionObserver` when scrolled into view.
- **DPR Capping**: Clamped to `dpr={[1, 1.7]}` to prevent GPU thermal throttling on high-resolution Retina displays.
- **Graceful WebGL Fallback**: When WebGL acceleration is unavailable (tested via headless flags), system renders a dedicated static research badge (`3D graphics acceleration unavailable. STATIC DIGITAL TWIN MODE`) without throwing runtime errors or blanking the page.

---

## 4. Accessibility & Motion Compliance
- **Reduced Motion**: Full support for `prefers-reduced-motion: reduce` (`useReducedMotion` hook disables rotor spin and eases transitions).
- **Focus Rings**: High-contrast vermilion focus outlines (`outline: 2px solid var(--signal)`) on interactive elements.
- **Text Contrast**: All technical labels and body copy comply with WCAG AA contrast standards against `#101411` and `#F3F1EB`.
