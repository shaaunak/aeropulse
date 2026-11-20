# Model Card: AeroPulse Causal LSTM (`model_lstm_v1`)

## Model Details
- **Architecture**: Causal Sequence-to-One 2-Layer Recurrent Neural Network (LSTM) with Dropout and Fully Connected Head.
- **Input Dimension**: $30 \times 17$ (30 historical operating cycles $\times$ 17 normalized features).
- **Hidden Dimensions**: 64 recurrent hidden units per layer, 32 units in dense projection head.
- **Dropout Rate**: 0.20 between recurrent layers and preceding the output projection.
- **Parameters**: 55,233 trainable parameters.
- **Training Objective**: Mean Squared Error (MSE) on piecewise-linear capped RUL targets.
- **Framework**: PyTorch 2.15.0 on Apple Silicon MPS (Metal Performance Shaders).
- **Date**: November 2026.

## Intended Use
- **Primary Use**: Prognostic research and degradation trajectory modeling on NASA C-MAPSS turbofan simulation data.
- **Out-of-Scope**: This model is **NOT certified for actual flight safety, civil aviation airworthiness, or real-time flight operations**. It is strictly a benchmarked research system trained on synthetic C-MAPSS simulator runs.

## Performance Metrics
- **Dataset**: NASA C-MAPSS FD001 Test Set ($N=100$ engines).
- **Official Uncapped Test RMSE**: **15.7545** cycles.
- **Official Uncapped Test MAE**: **10.5787** cycles.
- **Official NASA Asymmetric Score**: **418.66**.
- **Capped Test RMSE (125c)**: **14.3881** cycles.

## Ethical & Safety Considerations
- **Non-Endorsement**: AeroPulse is an independent academic research project and is not endorsed or certified by NASA, the FAA, or any aircraft propulsion manufacturer.
- **Transparent Uncertainty**: The application explicitly reports prediction residuals and error bounds, avoiding deceptive 100% confidence claims or arbitrary 'green safe-to-fly' badges.
