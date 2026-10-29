import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_2param_log_linear(x_data, y_data):
    """
    Fits 2-parameter power law: L = a * X^(-alpha)
    Using ordinary linear regression in log-space: ln(L) = ln(a) - alpha * ln(X)
    WITHOUT parameter bounds as requested.
    Returns dict with alpha, a, r2_log, r2_lin, fit_fn, label
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept = np.polyfit(log_x, log_y, 1)
    alpha = -slope
    a = np.exp(intercept)

    # R^2 in log space
    pred_log_y = slope * log_x + intercept
    ss_tot_log = np.sum((log_y - np.mean(log_y)) ** 2)
    ss_res_log = np.sum((log_y - pred_log_y) ** 2)
    r2_log = 1.0 - (ss_res_log / ss_tot_log) if ss_tot_log > 0 else 0.0

    # R^2 in linear space
    pred_y = a * np.power(x, -alpha)
    ss_tot_lin = np.sum((y - np.mean(y)) ** 2)
    ss_res_lin = np.sum((y - pred_y) ** 2)
    r2_lin = 1.0 - (ss_res_lin / ss_tot_lin) if ss_tot_lin > 0 else 0.0

    fit_fn = lambda val: a * np.power(val, -alpha)
    label = f"$L = {a:.2f} \\cdot X^{{-{alpha:.3f}}}$ ($R^2={r2_log:.4f}$)"

    return {
        "a": float(a),
        "alpha": float(alpha),
        "r2_log": float(r2_log),
        "r2_lin": float(r2_lin),
        "fit_fn": fit_fn,
        "label": label,
        "type": "2-param"
    }

def power_law_3param(x, a, alpha, c):
    return a * np.power(x, -alpha) + c

def fit_3param_power_law(x_data, y_data):
    """
    Fits 3-parameter power law: L = a * X^(-alpha) + c
    Only executed if len(x_data) >= 5.
    Reports if any parameter touches boundaries.
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    if len(x) < 5:
        return None

    # Bounds: a in [1e-6, 1e4], alpha in [1e-4, 5.0], c in [0.0, min(y)]
    lower_bounds = [1e-6, 1e-4, 0.0]
    upper_bounds = [1e4, 5.0, float(np.min(y))]

    try:
        # Initialize from 2-param fit
        fit2 = fit_2param_log_linear(x, y)
        p0 = [max(fit2["a"] * 0.8, 1e-5), max(fit2["alpha"], 0.01), float(np.min(y)) * 0.5]

        popt, pcov = curve_fit(
            power_law_3param, x, y,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )
        a, alpha, c = popt

        # Check boundary touch (within 1% or 1e-4 of bounds)
        boundary_flags = []
        tol = 1e-3
        if abs(a - lower_bounds[0]) < tol or abs(a - upper_bounds[0]) / upper_bounds[0] < tol:
            boundary_flags.append(f"a chạm biên ({a:.4g})")
        if abs(alpha - lower_bounds[1]) < tol or abs(alpha - upper_bounds[1]) < tol:
            boundary_flags.append(f"alpha chạm biên ({alpha:.4f})")
        if abs(c - lower_bounds[2]) < tol or abs(c - upper_bounds[2]) < tol:
            boundary_flags.append(f"c chạm biên ({c:.4f})")

        # R^2 in linear space
        pred_y = power_law_3param(x, *popt)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - pred_y) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        fit_fn = lambda val: power_law_3param(val, *popt)
        boundary_note = f" [CẢNH BÁO: {', '.join(boundary_flags)}]" if boundary_flags else ""
        label = f"$L = {a:.2f} \\cdot X^{{-{alpha:.3f}}} + {c:.3f}$ ($R^2={r2:.4f}$){boundary_note}"

        return {
            "a": float(a),
            "alpha": float(alpha),
            "c": float(c),
            "r2": float(r2),
            "boundary_flags": boundary_flags,
            "fit_fn": fit_fn,
            "label": label,
            "type": "3-param"
        }
    except Exception as e:
        print(f"3-parameter fit failed: {e}")
        return None

def plot_model_scaling(df, output_path="scaling_law_model_size.png"):
    # Filter 10,000 instances
    sub_df = df[df["train_instances"] == 10000].copy()
    if sub_df.empty:
        print("No 10000 instances data for model scaling.")
        return

    # Group by model_name to get mean, std across seeds
    grouped = sub_df.groupby("model_name").agg({
        "non_emb_params": "first",
        "eval_loss": ["mean", "std", "count"]
    }).reset_index()

    grouped.columns = ["model_name", "non_emb_params", "loss_mean", "loss_std", "count"]
    grouped = grouped.sort_values(by="non_emb_params")

    # If std is NaN (e.g. single run), set to 0.0
    grouped["loss_std"] = grouped["loss_std"].fillna(0.0)

    x = grouped["non_emb_params"].values
    y = grouped["loss_mean"].values
    yerr = grouped["loss_std"].values
    names = grouped["model_name"].values

    plt.figure(figsize=(10, 6.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Error bars
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='o', color="#1f77b4", ecolor="#e74c3c", elinewidth=2, capsize=6, capthick=2,
        markersize=9, zorder=5, label=f"Thực nghiệm (Mean ± 1 SD qua {int(grouped['count'].max())} seeds)"
    )

    # Annotate models
    for xi, yi, y_sd, name in zip(x, y, yerr, names):
        short_name = name.split("/")[-1].replace("bert_uncased_", "")
        txt = f" {short_name}\n {xi/1e6:.2f}M\n ({yi:.4f}±{y_sd:.4f})" if y_sd > 0 else f" {short_name}\n {xi/1e6:.2f}M\n ({yi:.4f})"
        plt.annotate(
            txt,
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#2c3e50',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
        )

    # 1. Fit 2-param log-linear (no bounds)
    fit2 = fit_2param_log_linear(x, y)
    x_fit = np.geomspace(min(x) * 0.7, max(x) * 1.4, 300)
    y_fit2 = fit2["fit_fn"](x_fit)
    plt.plot(x_fit, y_fit2, color="#2980b9", linestyle="--", linewidth=2.2,
             label=f"Fit 2-tham số: {fit2['label']}")

    # 2. Fit 3-param (only if >= 5 points)
    fit3 = fit_3param_power_law(x, y)
    if fit3 is not None:
        y_fit3 = fit3["fit_fn"](x_fit)
        plt.plot(x_fit, y_fit3, color="#e67e22", linestyle="-.", linewidth=2.0,
                 label=f"Fit 3-tham số: {fit3['label']}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Số tham số Non-Embedding ($N$) [Log-scale]", fontsize=12, fontweight='bold')
    plt.ylabel("Contrastive Perplexity Loss ($L$) [Log-scale]", fontsize=12, fontweight='bold')
    plt.title(f"Quy luật Lũy thừa Kích thước Mô hình (Model Scaling Law)\n(MS MARCO Passage, D = 10,000 mẫu, 3 epochs, lr = 1e-4)",
              fontsize=13, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor="white", framealpha=0.92, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved model scaling plot with error bars to {output_path}")

    return {
        "fit2": fit2,
        "fit3": fit3,
        "grouped": grouped
    }

def plot_data_scaling(df, target_model="google/bert_uncased_L-4_H-512_A-8", output_path="scaling_law_data_size.png"):
    sub_df = df[df["model_name"] == target_model].copy()
    if sub_df.empty:
        print(f"No data for target model {target_model}")
        return

    # Group by train_instances
    grouped = sub_df.groupby("train_instances").agg({
        "eval_loss": ["mean", "std", "count"]
    }).reset_index()

    grouped.columns = ["train_instances", "loss_mean", "loss_std", "count"]
    grouped = grouped.sort_values(by="train_instances")
    grouped["loss_std"] = grouped["loss_std"].fillna(0.0)

    if len(grouped) < 2:
        print("Not enough data sizes to plot data scaling.")
        return

    x = grouped["train_instances"].values
    y = grouped["loss_mean"].values
    yerr = grouped["loss_std"].values

    plt.figure(figsize=(10, 6.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Error bars
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='s', color="#27ae60", ecolor="#e74c3c", elinewidth=2, capsize=6, capthick=2,
        markersize=9, zorder=5, label=f"Thực nghiệm (Mean ± 1 SD qua {int(grouped['count'].max())} seeds)"
    )

    for xi, yi, y_sd in zip(x, y, yerr):
        txt = f" D={xi:,}\n ({yi:.4f}±{y_sd:.4f})" if y_sd > 0 else f" D={xi:,}\n ({yi:.4f})"
        plt.annotate(
            txt,
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#2c3e50',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
        )

    # 1. Fit 2-param log-linear (no bounds)
    fit2 = fit_2param_log_linear(x, y)
    x_fit = np.geomspace(min(x) * 0.7, max(x) * 1.4, 300)
    y_fit2 = fit2["fit_fn"](x_fit)
    plt.plot(x_fit, y_fit2, color="#16a085", linestyle="--", linewidth=2.2,
             label=f"Fit 2-tham số: $L = {fit2['a']:.2f} \\cdot D^{{-{fit2['alpha']:.3f}}}$ ($R^2={fit2['r2_log']:.4f}$)")

    # 2. Fit 3-param (only if >= 5 points)
    fit3 = fit_3param_power_law(x, y)
    if fit3 is not None:
        y_fit3 = fit3["fit_fn"](x_fit)
        plt.plot(x_fit, y_fit3, color="#d35400", linestyle="-.", linewidth=2.0,
                 label=f"Fit 3-tham số: $L = {fit3['a']:.2f} \\cdot D^{{-{fit3['alpha']:.3f}}} + {fit3['c']:.3f}$ ($R^2={fit3['r2']:.4f}$)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Số lượng Mẫu Huấn luyện ($D$) [Log-scale]", fontsize=12, fontweight='bold')
    plt.ylabel("Contrastive Perplexity Loss ($L$) [Log-scale]", fontsize=12, fontweight='bold')
    plt.title(f"Quy luật Lũy thừa Lượng Dữ liệu (Data Scaling Law)\n(Mô hình: BERT-small L-4_H-512_A-8, 3 epochs, lr = 1e-4)",
              fontsize=13, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor="white", framealpha=0.92, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved data scaling plot with error bars to {output_path}")

    return {
        "fit2": fit2,
        "fit3": fit3,
        "grouped": grouped
    }

def generate_all_plots(csv_path="results.csv", output_dir="."):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    m_info = plot_model_scaling(df, os.path.join(output_dir, "scaling_law_model_size.png"))
    d_info = plot_data_scaling(df, output_path=os.path.join(output_dir, "scaling_law_data_size.png"))
    return m_info, d_info

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    generate_all_plots(csv_file, out_dir)
