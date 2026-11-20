# AeroPulse Methodology & Leakage Prevention Protocol

## 1. Problem Formulation
Turbofan remaining useful life (RUL) estimation is formulated as a sequence-to-one multivariate regression problem:
$$f: \mathbf{X}_{t-W+1:t} \in \mathbb{R}^{W \times D} \longrightarrow \hat{y}_t \in \mathbb{R}_{\ge 0}$$
where:
- $W = 30$ is the causal observation window size in operating cycles.
- $D = 17$ is the number of active, non-constant feature channels (15 sensors + 2 operational settings).
- $t$ is the current evaluation cycle endpoint.
- $\hat{y}_t$ is the predicted remaining cycles before HPC degradation induces system failure.

---

## 2. Leakage Safeguards & Split Architecture

Predictive maintenance datasets are highly vulnerable to subtle forms of data leakage that invalidate real-world deployment claims:

### A. Engine-Disjoint Splitting
- **Vulnerability**: Splitting individual cycles or time slices randomly across train and test causes autocorrelation leakage, where the model memorizes trajectory trajectories of the same physical unit.
- **Protocol**: Train engines are split **by unit ID** into an 80/20 train/validation split (80 engines for model training, 20 engines for validation and early stopping). Test engines ($N=100$) are entirely distinct physical units provided in `test_FD001.txt`.

### B. Train-Only Feature Selection & Scaler Fitting
- **Vulnerability**: Computing dataset statistics (mean, variance, min, max) across the full dataset leaks test distribution information into model parameters.
- **Protocol**: 
  - Variances were computed **exclusively on the 80 training engines**.
  - 7 features exhibited zero or near-zero variance ($\sigma^2 < 10^{-12}$): `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`. These were dropped from all splits.
  - 17 features were retained.
  - `StandardScaler` parameters ($\mu, \sigma$) were fitted **solely on the 80 training engines** and persisted in the model checkpoint.

### C. Strict Intra-Engine Causal Windowing
- **Vulnerability**: Sliding windows crossing engine boundaries leak end-of-life state from engine $k$ into the start of engine $k+1$.
- **Protocol**: Sliding windows are strictly constrained within each engine's trajectory. For engines with fewer than $W=30$ cycles, causal left-edge repeat padding is applied to preserve sequence alignment without lookahead.

### D. Single Endpoint Official Evaluation
- **Protocol**: In NASA C-MAPSS FD001, test sequences terminate prior to failure. The official test set provides exactly one ground truth RUL per engine at its final observed cycle. Evaluation evaluates **exactly one** prediction per test engine from its final window against `RUL_FD001.txt`.

---

## 3. RUL Target Construction & Capping

In the training split, engines operate until complete functional failure:
$$\text{RUL}_{\text{raw}}(c) = C_{\text{max}} - c$$
To stabilize early training where degradation has not yet initiated, the training target is piecewise-capped at $R_{\text{cap}} = 125$ cycles:
$$\text{RUL}_{\text{target}}(c) = \min(\text{RUL}_{\text{raw}}(c), 125.0)$$
In official benchmark testing, predictions are compared directly against the **uncapped official test ground truth** to ensure genuine comparability with published PHM literature.
