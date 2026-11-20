# AeroPulse Attributions & Dataset Provenance

## 1. Dataset Citation
The data used in AeroPulse is the **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Jet Engine Degradation Simulation Dataset**, curated and published by NASA Prognostics Center of Excellence (PCoE):

> **Citation**:
> A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, Oct 2008.

- **Data Portal URL**: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- **Repository URL**: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## 2. 3D Digital Twin & Procedural Geometry
The 3D procedural turbofan visualization in AeroPulse was designed and implemented procedurally in React Three Fiber / three.js.
- Geometric representation: Artistic cutaway and bladed rotary assembly.
- Not a CAD replica: The geometry is an educational visualization and does not represent an ITAR-controlled propulsion architecture.
- Sensor callouts: Sensor grouping mappings are illustrative and conceptual; NASA C-MAPSS channels are anonymized and do not specify exact mounting topology.

## 3. Libraries & Open-Source Stack
- **PyTorch**: Deep learning framework (`torch.nn.LSTM`).
- **Scikit-learn**: Baseline regressors, variance filtering, and standard scaling.
- **FastAPI & Uvicorn**: High-performance RESTful API and OpenAPI generation.
- **React, Vite & TypeScript**: Modern frontend application framework.
- **Three.js, React Three Fiber & Drei**: 3D WebGL rendering engine.
- **Lucide Icons**: Crisp interface iconography.
