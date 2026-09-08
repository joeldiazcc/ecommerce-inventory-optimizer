"""Utilidades de visualización."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from inventario_ecommerce import config


def plot_top_products_by_sales(
    summary: pd.DataFrame,
    top_n: int = 10,
    save: bool = True,
) -> Path | None:
    """Barras horizontales de los top N productos por ventas del último trimestre."""
    plot_df = summary.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df[config.COL_DESCRIPTION], plot_df["TotalSales"])
    ax.set_xlabel("Ventas totales")
    ax.set_ylabel("Producto")
    ax.set_title(f"Top {top_n} productos — ventas último trimestre")
    fig.tight_layout()

    out_path = None
    if save:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.FIGURES_DIR / "top_products_last_quarter.png"
        fig.savefig(out_path, dpi=120)
    plt.show()
    return out_path


def plot_class_a_mae_comparison(
    global_metrics: pd.DataFrame,
    save: bool = True,
) -> Path | None:
    """Barras MAE media móvil vs ETS en el holdout de clase A."""
    row = global_metrics.iloc[0]
    labels = ["Media móvil 30d", "ETS Holt-Winters"]
    values = [float(row["GlobalMAE_ma30"]), float(row["GlobalMAE_ets"])]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#6c757d", "#0d6efd"])
    ax.set_ylabel("MAE (ud/día)")
    ax.set_title("Backtest 30d — top clase A")
    ax.bar_label(bars, fmt="{:.2f}")
    fig.tight_layout()

    out_path = None
    if save:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.FIGURES_DIR / "class_a_ets_vs_baseline.png"
        fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
