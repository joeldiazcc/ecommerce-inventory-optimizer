"""Predicción baseline y política de reorden por SKU."""

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


def forecast_30d_baseline(
    daily_sku_demand: pd.DataFrame,
    lookback_days: int = 30,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Forecast simple por SKU: media diaria de la ventana lookback."""
    max_date = daily_sku_demand["Date"].max()
    hist = daily_sku_demand[daily_sku_demand["Date"] > (max_date - pd.Timedelta(days=lookback_days))]

    forecast = (
        hist.groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .agg(forecast_daily=("QuantitySold", "mean"))
        .fillna(0.0)
    )
    forecast["forecast_30d"] = forecast["forecast_daily"] * horizon_days
    return forecast


def build_reorder_policy(
    latest_features: pd.DataFrame,
    forecast: pd.DataFrame,
    abc: pd.DataFrame,
    lead_time_days: int = 14,
    review_period_days: int = 7,
) -> pd.DataFrame:
    """Construye tabla de punto de reorden y stock sugerido."""
    policy = forecast[[config.COL_STOCK_CODE, config.COL_DESCRIPTION, "forecast_daily", "forecast_30d"]].merge(
        latest_features,
        on=[config.COL_STOCK_CODE, config.COL_DESCRIPTION],
        how="inner",
    ).merge(
        abc[[config.COL_STOCK_CODE, config.COL_DESCRIPTION, "ABCClass", "TotalSales"]],
        on=[config.COL_STOCK_CODE, config.COL_DESCRIPTION],
        how="left",
    )

    policy["ABCClass"] = policy["ABCClass"].fillna("C")

    # Nivel de servicio por clase ABC (simple y explicable para baseline junior)
    z_map = {"A": 1.88, "B": 1.65, "C": 1.28}
    policy["z_service"] = policy["ABCClass"].map(z_map).fillna(1.28)
    sigma = policy["demand_std_30d"].fillna(0.0)

    policy["safety_stock"] = policy["z_service"] * sigma * np.sqrt(lead_time_days)
    policy["lead_time_demand"] = policy["forecast_daily"] * lead_time_days
    policy["reorder_point"] = policy["lead_time_demand"] + policy["safety_stock"]
    policy["target_stock"] = policy["reorder_point"] + (
        policy["forecast_daily"] * review_period_days
    )

    policy["recommended_order_qty"] = (
        policy["target_stock"] - policy["reorder_point"]
    ).clip(lower=0.0)
    policy["recommendation"] = np.where(
        policy["ABCClass"] == "A",
        "Monitoreo diario; evitar quiebres",
        np.where(
            policy["ABCClass"] == "B",
            "Revisión semanal",
            "Revisión quincenal",
        ),
    )

    keep_cols = [
        config.COL_STOCK_CODE,
        config.COL_DESCRIPTION,
        "ABCClass",
        "TotalSales",
        "forecast_daily",
        "forecast_30d",
        "demand_mean_30d",
        "demand_std_30d",
        "demand_cv_30d",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order_qty",
        "recommendation",
    ]
    return policy[keep_cols].sort_values("forecast_30d", ascending=False).reset_index(drop=True)


def predict() -> pd.DataFrame:
    """Genera tabla final de recomendaciones de inventario por SKU."""
    raw = load_transactions()
    clean = clean_transactions(raw)
    daily = build_daily_sku_demand(clean)

    rolling = build_rolling_features(daily)
    latest_features = (
        rolling.sort_values("Date")
        .groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    forecast = forecast_30d_baseline(daily, lookback_days=30, horizon_days=30)
    abc = compute_abc_classification(sales_by_product_last_quarter(clean))

    policy = build_reorder_policy(latest_features, forecast, abc)
    save_processed(policy, "inventory_reorder_recommendations.csv")
    return policy


if __name__ == "__main__":
    recommendations = predict()
    print("Predicción baseline y política de reorden completadas.")
    print(f"SKUs recomendados: {len(recommendations):,}")
    print(f"Salida: {config.PROCESSED_DATA_DIR / 'inventory_reorder_recommendations.csv'}")
