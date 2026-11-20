"""Generate publication-grade diagnostic plots for AeroPulse research documentation."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apply clean minimalist aerospace styling
plt.rcParams.update({
    "font.sans-serif": "Helvetica, Arial, sans-serif",
    "font.family": "sans-serif",
    "figure.facecolor": "#0D110F",
    "axes.facecolor": "#121715",
    "axes.edgecolor": "#2D3732",
    "axes.labelcolor": "#E5E5DF",
    "xtick.color": "#9E9E96",
    "ytick.color": "#9E9E96",
    "text.color": "#E5E5DF",
    "grid.color": "#202824",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "#171D1A",
    "legend.edgecolor": "#2D3732",
})

out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

with open("artifacts/metrics/evaluation_summary.json") as f:
    eval_data = json.load(f)

y_true = np.array(eval_data["ground_truth_test_rul"])
lstm_pred = np.array(eval_data["models"]["lstm"]["predictions"])
gru_pred = np.array(eval_data["models"]["gru"]["predictions"])
ridge_pred = np.array(eval_data["models"]["ridge_tabular"]["predictions"])

errors_lstm = lstm_pred - y_true
errors_gru = gru_pred - y_true

# -------------------------------------------------------------
# 1. Error vs True RUL (with Near-End-of-Life highlighting)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.axhline(0, color="#FF5E39", linestyle="-", linewidth=1.5, alpha=0.9, label="Zero Error (Perfect)")
ax.axvline(30, color="#DAEFAB", linestyle=":", linewidth=1.2, label="Critical Near-EOL (True RUL <= 30)")
ax.axvspan(0, 30, color="#DAEFAB", alpha=0.07)

ax.scatter(y_true, errors_lstm, color="#5EEAD4", alpha=0.85, s=36, edgecolors="#101411", linewidths=0.5, label=f"LSTM (RMSE: {eval_data['models']['lstm']['official_test_uncapped']['rmse']:.2f})")
ax.scatter(y_true, errors_gru, color="#F59E0B", alpha=0.6, s=28, marker="^", edgecolors="#101411", linewidths=0.5, label=f"GRU (RMSE: {eval_data['models']['gru']['official_test_uncapped']['rmse']:.2f})")

ax.set_title("Prediction Error ($e_i = \\hat{y}_i - y_i$) vs True RUL", fontsize=12, pad=12, fontweight="bold")
ax.set_xlabel("Official True Remaining Useful Life (Cycles)", fontsize=10)
ax.set_ylabel("Error: Overestimate (+) / Underestimate (-)", fontsize=10)
ax.grid(True)
ax.legend(frameon=True, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(out_dir / "error_vs_true_rul.png")
plt.close(fig)
print("[AeroPulse] Saved docs/figures/error_vs_true_rul.png")

# -------------------------------------------------------------
# 2. Residual Distribution Histogram & Density
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
bins = np.linspace(-50, 50, 25)

ax.hist(errors_lstm, bins=bins, color="#5EEAD4", alpha=0.65, edgecolor="#2D3732", label=f"LSTM (Mean: {np.mean(errors_lstm):.2f}, Std: {np.std(errors_lstm):.2f})")
ax.hist(errors_gru, bins=bins, color="#F59E0B", alpha=0.45, edgecolor="#2D3732", label=f"GRU (Mean: {np.mean(errors_gru):.2f}, Std: {np.std(errors_gru):.2f})")
ax.axvline(0, color="#FF5E39", linestyle="--", linewidth=1.5, label="Unbiased Zero")

ax.set_title("Test Prediction Residual Distribution ($N=100$ Test Engines)", fontsize=12, pad=12, fontweight="bold")
ax.set_xlabel("Prediction Error $e_i$ (Cycles)", fontsize=10)
ax.set_ylabel("Engine Count", fontsize=10)
ax.grid(True)
ax.legend(frameon=True, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(out_dir / "residual_distribution.png")
plt.close(fig)
print("[AeroPulse] Saved docs/figures/residual_distribution.png")

# -------------------------------------------------------------
# 3. Model Benchmark Multi-Seed Boxplot
# -------------------------------------------------------------
if Path("artifacts/experiments/multi_seed_benchmark.json").exists():
    with open("artifacts/experiments/multi_seed_benchmark.json") as f:
        exp_data = json.load(f)

    lstm_rmses = [r["test_rmse"] for r in exp_data["lstm"]["runs"]]
    gru_rmses = [r["test_rmse"] for r in exp_data["gru"]["runs"]]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bp = ax.boxplot([lstm_rmses, gru_rmses], tick_labels=["Causal LSTM", "Causal GRU"], patch_artist=True, widths=0.4)
    bp['boxes'][0].set_facecolor("#5EEAD4")
    bp['boxes'][1].set_facecolor("#F59E0B")
    for box in bp['boxes']:
        box.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color("#FF5E39")
        median.set_linewidth(2)

    ax.set_title(f"5-Seed Test RMSE Distribution Across Random Initializations", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Official Test RMSE (Cycles)", fontsize=10)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_multiseed.png")
    plt.close(fig)
    print("[AeroPulse] Saved docs/figures/model_comparison_multiseed.png")

# -------------------------------------------------------------
# 4. Critical Near-EOL Error Analysis (< 30 cycles remaining)
# -------------------------------------------------------------
near_eol_mask = y_true <= 30
eol_true = y_true[near_eol_mask]
eol_lstm_err = errors_lstm[near_eol_mask]
eol_gru_err = errors_gru[near_eol_mask]

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
engine_indices = np.arange(1, len(eol_true) + 1)
bar_width = 0.35

ax.bar(engine_indices - bar_width/2, eol_lstm_err, width=bar_width, color="#5EEAD4", alpha=0.8, label=f"LSTM Near-EOL MAE: {np.mean(np.abs(eol_lstm_err)):.2f}c")
ax.bar(engine_indices + bar_width/2, eol_gru_err, width=bar_width, color="#F59E0B", alpha=0.8, label=f"GRU Near-EOL MAE: {np.mean(np.abs(eol_gru_err)):.2f}c")
ax.axhline(0, color="#FF5E39", linestyle="-", linewidth=1.2)

ax.set_title(f"Near-End-of-Life Error Breakdown (Engines with True RUL $\\leq$ 30, $n={len(eol_true)}$)", fontsize=12, pad=12, fontweight="bold")
ax.set_xlabel("Engine Test Index (Critical Subfleet)", fontsize=10)
ax.set_ylabel("Prediction Residual (Cycles)", fontsize=10)
ax.set_xticks(engine_indices)
ax.grid(True, axis="y")
ax.legend(frameon=True, fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "near_eol_error_analysis.png")
plt.close(fig)
print("[AeroPulse] Saved docs/figures/near_eol_error_analysis.png")
