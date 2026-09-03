"""Entrenamiento baseline y validación temporal para demanda por SKU."""

from __future__ import annotations

import numpy as np
import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.dataset import load_transactions, save_processed
from inventario_ecommerce.features import (
    build_daily_sku_demand,
    build_rolling_features,
    clean_transactions,
    compute_abc_classification,
    sales_by_product_last_quarter,
)


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = y_true.replace(0, np.nan)
    mape = ((y_true - y_pred).abs() / denom).replace([np.inf, -np.inf], np.nan).dropna()
    if mape.empty:
        return float("nan")
    return float(mape.mean() * 100)


def temporal_backtest_baseline(
    daily_sku_demand: pd.DataFrame,
    horizon_days: int = 30,
    lookback_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest temporal por SKU usando baseline de media móvil simple."""
    max_date = daily_sku_demand["Date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon_days)

    train = daily_sku_demand[daily_sku_demand["Date"] <= cutoff].copy()
    test = daily_sku_demand[daily_sku_demand["Date"] > cutoff].copy()

    if train.empty or test.empty:
        raise ValueError("No hay suficientes datos para backtest temporal.")

    tail = train[train["Date"] > (cutoff - pd.Timedelta(days=lookback_days))]
    sku_daily_mean = (
        tail.groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .agg(pred_daily_qty=("QuantitySold", "mean"))
        .fillna(0.0)
    )

    test_eval = test.merge(
        sku_daily_mean,
        on=[config.COL_STOCK_CODE, config.COL_DESCRIPTION],
        how="left",
    )
    test_eval["pred_daily_qty"] = test_eval["pred_daily_qty"].fillna(0.0)
    test_eval["abs_error"] = (test_eval["QuantitySold"] - test_eval["pred_daily_qty"]).abs()

    metrics = (
        test_eval.groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .agg(
            MAE=("abs_error", "mean"),
            RealMeanDaily=("QuantitySold", "mean"),
            PredMeanDaily=("pred_daily_qty", "mean"),
            DaysEval=("Date", "nunique"),
        )
    )
    metrics["MAPE"] = metrics.apply(
        lambda r: np.nan
        if r["RealMeanDaily"] == 0
        else abs(r["RealMeanDaily"] - r["PredMeanDaily"]) / r["RealMeanDaily"] * 100,
        axis=1,
    )

    global_metrics = pd.DataFrame(
        [
            {
                "CutoffDate": cutoff.date().isoformat(),
                "EvalStartDate": (cutoff + pd.Timedelta(days=1)).date().isoformat(),
                "EvalEndDate": max_date.date().isoformat(),
                "GlobalMAE": float(test_eval["abs_error"].mean()),
                "GlobalMAPE": _safe_mape(
                    test_eval["QuantitySold"],
                    test_eval["pred_daily_qty"],
                ),
                "HorizonDays": horizon_days,
                "LookbackDays": lookback_days,
            }
        ]
    )
    return metrics, global_metrics


def train() -> dict[str, pd.DataFrame]:
    """Ejecuta pipeline de entrenamiento baseline y guarda artefactos."""
    raw = load_transactions()
    clean = clean_transactions(raw)

    last_quarter_sales = sales_by_product_last_quarter(clean)
    abc = compute_abc_classification(last_quarter_sales, sales_col="TotalSales")
    save_processed(abc, "abc_last_quarter.csv")

    daily = build_daily_sku_demand(clean)
    rolling = build_rolling_features(daily)
    latest_features = (
        rolling.sort_values("Date")
        .groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    save_processed(latest_features, "sku_rolling_features_latest.csv")

    sku_metrics, global_metrics = temporal_backtest_baseline(daily)
    save_processed(sku_metrics, "forecast_backtest_by_sku.csv")
    save_processed(global_metrics, "forecast_backtest_global.csv")

    return {
        "abc": abc,
        "latest_features": latest_features,
        "backtest_by_sku": sku_metrics,
        "backtest_global": global_metrics,
    }


if __name__ == "__main__":
    outputs = train()
    print("Entrenamiento baseline completado.")
    print("Artefactos:")
    for key, value in outputs.items():
        print(f"- {key}: {len(value):,} filas")
    print(f"Carpeta de salida: {config.PROCESSED_DATA_DIR}")
