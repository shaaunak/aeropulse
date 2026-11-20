# Model Card: AeroPulse Causal LSTM & GRU Trajectory Models

## 1. Model Details
- **Architecture**: Causal Sequence-to-One 2-Layer Recurrent Neural Network (LSTM / GRU) with intermediate dropout and fully connected linear head.
- **Input Dimension**: 30 historical operating cycles x 17 normalized features (15 aerothermal sensor channels + 2 operational settings).
- **Recurrent Hidden Dimensions**: 64 units per recurrent layer.
- **Dense Projection**: 64 -> 32 -> 1 with ReLU activation and non-negative clamping.
- **Dropout Rate**: 0.20 between recurrent layers and preceding linear projection.
- **Trainable Parameters**: 55,233 (LSTM) / 42,881 (GRU).
- **Optimization Objective**: Mean Squared Error (MSE) on piecewise-linear capped RUL targets (125-cycle cap).
- **Framework & Hardware**: PyTorch 2.x on Apple Silicon MPS & CPU.
- **Development Date**: November 2026.

## 2. Intended Use & Boundaries
- **Research Scope**: Degradation trajectory analysis, sequence-to-one regression, and remaining useful life (RUL) estimation on the NASA C-MAPSS FD001 benchmark.
- **Out-of-Scope**: This software is **NOT certified for actual aircraft airworthiness, real-time flight operations, or flight dispatch**. It is strictly a benchmarked research application trained on synthetic engine simulator runs.

## 3. Model Selection & Multi-Seed Validation
- **Selection Protocol**: Models are selected based on cross-validation on an 80/20 engine-disjoint development split (80 training engines, 20 validation engines). The held-out official test set (100 engines) is never used for architecture tuning or early stopping.
- **Multi-Seed Benchmark (5 Seeds: 42, 123, 456, 789, 1011)**:
  - **Causal LSTM Test RMSE (Uncapped)**: 15.66 +/- 0.34 cycles | **NASA Score**: 447.41 +/- 80.24
  - **Causal GRU Test RMSE (Uncapped)**: 29.70 +/- 31.58 cycles | **NASA Score**: 112,175.54 +/- 249,792.38
- **Statistical Findings**: While GRU can achieve lower RMSE on specific runs, LSTM demonstrates significantly higher optimization stability across random initializations. GRU exhibited optimization failure (gradient divergence) on seed 456 (Test RMSE: 86.20), whereas LSTM consistently converged with tight variance (sigma = 0.34 cycles).

## 4. Evaluation Metrics & Asymmetric Penalty
- **Root Mean Squared Error (RMSE)**: sqrt(mean((y_pred - y_true)^2))
- **Mean Absolute Error (MAE)**: mean(|y_pred - y_true|)
- **NASA PHM08 Asymmetric Score**:
  - Let e_i = y_pred_i - y_true_i
  - If e_i < 0 (underestimate / conservative call): s_i = exp(-e_i / 13) - 1
  - If e_i >= 0 (overestimate / dangerous late call): s_i = exp(e_i / 10) - 1
  - Score = sum(s_i) over all 100 test engines.

## 5. Limitations & Ethical Considerations
- **Simulated Data**: The NASA C-MAPSS dataset represents computer simulation data from the Commercial Modular Aero-Propulsion System Simulation package. It contains idealized noise and failure modes not representative of all real-world environmental factors (e.g., volcanic ash, bird strikes, foreign object damage).
- **Uncertainty Bounds**: Point predictions do not account for epistemic uncertainty without conformal prediction or Bayesian ensembling. Near-EOL prediction confidence intervals should be interpreted as empirical error distributions.
