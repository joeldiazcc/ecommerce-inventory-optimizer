"""Predicción baseline y política de reorden por SKU."""

from __future__ import annotations

import numpy as np
import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.dataset import (
    load_stock_positions,
    load_transactions,
    save_processed,
)
from inventario_ecommerce.features import (
    build_rolling_features,
    clean_transactions,
    compute_abc_classification,
    prepare_daily_demand,
    sales_by_product_last_quarter,
)
from inventario_ecommerce.modeling.ets import forecast_ets_class_a, overlay_ets_forecast


def forecast_30d_baseline(
    daily_sku_demand: pd.DataFrame,
    lookback_days: int = 30,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Forecast simple por SKU: media diaria de la ventana lookback.

    Solo incluye SKUs con al menos una unidad vendida en el lookback
    (tras rellenar ceros, la media ya refleja días sin venta).
    """
    max_date = daily_sku_demand["Date"].max()
    hist = daily_sku_demand[
        daily_sku_demand["Date"] > (max_date - pd.Timedelta(days=lookback_days))
    ]

    forecast = (
        hist.groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .agg(
            forecast_daily=("QuantitySold", "mean"),
            lookback_qty=("QuantitySold", "sum"),
        )
        .fillna(0.0)
    )
    forecast = forecast[forecast["lookback_qty"] > 0].drop(columns=["lookback_qty"])
    forecast["forecast_30d"] = forecast["forecast_daily"] * horizon_days
    forecast["forecast_model"] = "ma30"
    return forecast.reset_index(drop=True)


def _apply_stock_positions(
    policy: pd.DataFrame,
    stock_positions: pd.DataFrame | None,
    review_period_days: int,
) -> pd.DataFrame:
    """Calcula el pedido sugerido con o sin posición de inventario conocida."""
    out = policy.copy()
    if stock_positions is None or stock_positions.empty:
        out["on_hand"] = np.nan
        out["on_order"] = np.nan
        out["inventory_position"] = np.nan
        out["recommended_order_qty"] = (out["forecast_daily"] * review_period_days).clip(
            lower=0.0
        )
        out["order_basis"] = "ciclo_revision"
        return out

    positions = stock_positions.copy()
    positions[config.COL_STOCK_CODE] = (
        positions[config.COL_STOCK_CODE].astype(str).str.strip()
    )
    out["_stock"] = out[config.COL_STOCK_CODE].astype(str).str.strip()
    out = out.merge(
        positions.rename(columns={config.COL_STOCK_CODE: "_stock"}),
        on="_stock",
        how="left",
    ).drop(columns=["_stock"])

    out["on_hand"] = out["on_hand"].fillna(0.0)
    out["on_order"] = out["on_order"].fillna(0.0)
    out["inventory_position"] = out["on_hand"] + out["on_order"]
    out["recommended_order_qty"] = (
        out["target_stock"] - out["inventory_position"]
    ).clip(lower=0.0)
    # Solo se pide cuando la posición ha caído al punto de reorden.
    out.loc[out["inventory_position"] > out["reorder_point"], "recommended_order_qty"] = 0.0
    out["order_basis"] = "target_menos_posicion"
    return out


def build_reorder_policy(
    latest_features: pd.DataFrame,
    forecast: pd.DataFrame,
    abc: pd.DataFrame,
    lead_time_days: int = config.LEAD_TIME_DAYS,
    review_period_days: int = config.REVIEW_PERIOD_DAYS,
    stock_positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye tabla de punto de reorden y stock sugerido.

    Con `stock_positions` el pedido es `target − on_hand − on_order`; sin ella,
    la demanda del ciclo de revisión.
    """
    forecast_cols = [config.COL_STOCK_CODE, config.COL_DESCRIPTION, "forecast_daily", "forecast_30d"]
    if "forecast_model" in forecast.columns:
        forecast_cols.append("forecast_model")
    policy = forecast[forecast_cols].merge(
        latest_features,
        on=[config.COL_STOCK_CODE, config.COL_DESCRIPTION],
        how="inner",
    ).merge(
        abc[[config.COL_STOCK_CODE, config.COL_DESCRIPTION, "ABCClass", "TotalSales"]],
        on=[config.COL_STOCK_CODE, config.COL_DESCRIPTION],
        how="inner",
    )

    # Nivel de servicio por clase ABC (simple y explicable para baseline junior)
    policy["z_service"] = (
        policy["ABCClass"]
        .map(config.SERVICE_Z_BY_CLASS)
        .fillna(config.DEFAULT_SERVICE_Z)
    )
    sigma = policy["demand_std_30d"].fillna(0.0)

    policy["safety_stock"] = policy["z_service"] * sigma * np.sqrt(lead_time_days)
    policy["lead_time_demand"] = policy["forecast_daily"] * lead_time_days
    policy["reorder_point"] = policy["lead_time_demand"] + policy["safety_stock"]
    policy["target_stock"] = policy["reorder_point"] + (
        policy["forecast_daily"] * review_period_days
    )

    policy = _apply_stock_positions(policy, stock_positions, review_period_days)
    policy["recommendation"] = np.where(
        policy["ABCClass"] == "A",
        "Monitoreo diario; evitar quiebres",
        np.where(
            policy["ABCClass"] == "B",
            "Revisión semanal",
            "Revisión quincenal",
        ),
    )

    if "forecast_model" not in policy.columns:
        policy["forecast_model"] = "ma30"

    keep_cols = [
        config.COL_STOCK_CODE,
        config.COL_DESCRIPTION,
        "ABCClass",
        "TotalSales",
        "forecast_daily",
        "forecast_30d",
        "forecast_model",
        "demand_mean_30d",
        "demand_std_30d",
        "demand_cv_30d",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "on_hand",
        "on_order",
        "inventory_position",
        "recommended_order_qty",
        "order_basis",
        "recommendation",
    ]
    return policy[keep_cols].sort_values("forecast_30d", ascending=False).reset_index(drop=True)


def predict() -> pd.DataFrame:
    """Genera tabla final de recomendaciones de inventario por SKU."""
    raw = load_transactions()
    clean = clean_transactions(raw)
    abc = compute_abc_classification(sales_by_product_last_quarter(clean))
    daily = prepare_daily_demand(clean, abc=abc)

    rolling = build_rolling_features(daily)
    latest_features = (
        rolling.sort_values("Date")
        .groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    forecast = forecast_30d_baseline(daily, lookback_days=30, horizon_days=30)
    ets = forecast_ets_class_a(daily, abc)
    forecast = overlay_ets_forecast(forecast, ets)

    policy = build_reorder_policy(
        latest_features, forecast, abc, stock_positions=load_stock_positions()
    )
    save_processed(policy, "inventory_reorder_recommendations.csv")
    return policy


if __name__ == "__main__":
    recommendations = predict()
    print("Predicción baseline y política de reorden completadas.")
    print(f"SKUs recomendados: {len(recommendations):,}")
    print(f"Salida: {config.PROCESSED_DATA_DIR / 'inventory_reorder_recommendations.csv'}")
