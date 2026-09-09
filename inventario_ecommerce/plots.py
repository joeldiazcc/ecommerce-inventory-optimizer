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
    return out_path


def plot_policy_service_tradeoff(
    summary: pd.DataFrame,
    save: bool = True,
) -> Path | None:
    """Stock medio sostenido frente al fill rate logrado por cada política."""
    ordered = summary.sort_values("fill_rate")
    x = ordered["fill_rate"] * 100
    y = ordered["avg_on_hand_units"] / 1000

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="#adb5bd", linewidth=1, zorder=2)
    ax.scatter(x, y, s=90, color="#0d6efd", zorder=3)
    # Etiquetas alternadas arriba/abajo: hay políticas casi superpuestas.
    for i, row in enumerate(ordered.itertuples(index=False)):
        ax.annotate(
            row.policy,
            (row.fill_rate * 100, row.avg_on_hand_units / 1000),
            textcoords="offset points",
            xytext=(0, 11) if i % 2 == 0 else (0, -19),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Fill rate (% de unidades servidas)")
    ax.set_ylabel("Stock medio en almacén (miles de ud)")
    ax.set_title("Coste del nivel de servicio — holdout 30 días")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()

    out_path = None
    if save:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.FIGURES_DIR / "policy_service_tradeoff.png"
        fig.savefig(out_path, dpi=120)
    return out_path


def plot_sku_forecast_comparison(
    comparison: pd.DataFrame,
    title: str,
    save: bool = True,
) -> Path | None:
    """Serie real del holdout frente a la media móvil y al ETS, para un SKU."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(comparison["Date"], comparison["real"], color="#212529", label="Real")
    ax.plot(
        comparison["Date"],
        comparison["ma30"],
        color="#6c757d",
        linestyle="--",
        label="Media móvil 30d",
    )
    ax.plot(comparison["Date"], comparison["ets"], color="#0d6efd", label="ETS")
    ax.set_ylabel("Unidades/día")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = None
    if save:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.FIGURES_DIR / "class_a_sku_forecast.png"
        fig.savefig(out_path, dpi=120)
    return out_path
