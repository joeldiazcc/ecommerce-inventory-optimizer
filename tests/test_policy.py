"""Tests de la política de reorden y del uso de stock on-hand."""

import numpy as np
import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.modeling.predict import build_reorder_policy


def _policy_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = {
        config.COL_STOCK_CODE: ["A1"],
        config.COL_DESCRIPTION: ["TOP"],
    }
    forecast = pd.DataFrame(
        {**keys, "forecast_daily": [10.0], "forecast_30d": [300.0], "forecast_model": ["ma30"]}
    )
    latest_features = pd.DataFrame(
        {**keys, "demand_mean_30d": [10.0], "demand_std_30d": [0.0], "demand_cv_30d": [0.0]}
    )
    abc = pd.DataFrame({**keys, "ABCClass": ["A"], "TotalSales": [1000.0]})
    return latest_features, forecast, abc


def test_order_falls_back_to_review_cycle_without_stock_source():
    latest_features, forecast, abc = _policy_frames()

    policy = build_reorder_policy(latest_features, forecast, abc)

    assert policy.loc[0, "reorder_point"] == 140.0
    assert policy.loc[0, "target_stock"] == 210.0
    assert policy.loc[0, "recommended_order_qty"] == 70.0
    assert policy.loc[0, "z_service"] == config.SERVICE_Z_BY_CLASS["A"]
    assert policy.loc[0, "order_basis"] == "ciclo_revision"
    assert np.isnan(policy.loc[0, "on_hand"])


def test_order_closes_the_gap_to_target_when_stock_is_known():
    latest_features, forecast, abc = _policy_frames()
    positions = pd.DataFrame(
        {config.COL_STOCK_CODE: ["A1"], "on_hand": [50.0], "on_order": [30.0]}
    )

    policy = build_reorder_policy(
        latest_features, forecast, abc, stock_positions=positions
    )

    assert policy.loc[0, "inventory_position"] == 80.0
    assert policy.loc[0, "recommended_order_qty"] == 130.0
    assert policy.loc[0, "order_basis"] == "target_menos_posicion"


def test_no_order_when_position_is_above_the_reorder_point():
    latest_features, forecast, abc = _policy_frames()
    positions = pd.DataFrame(
        {config.COL_STOCK_CODE: ["A1"], "on_hand": [200.0], "on_order": [0.0]}
    )

    policy = build_reorder_policy(
        latest_features, forecast, abc, stock_positions=positions
    )

    assert policy.loc[0, "recommended_order_qty"] == 0.0


def test_missing_sku_in_stock_source_is_treated_as_empty_shelf():
    latest_features, forecast, abc = _policy_frames()
    positions = pd.DataFrame(
        {config.COL_STOCK_CODE: ["OTRO"], "on_hand": [10.0], "on_order": [0.0]}
    )

    policy = build_reorder_policy(
        latest_features, forecast, abc, stock_positions=positions
    )

    assert policy.loc[0, "inventory_position"] == 0.0
    assert policy.loc[0, "recommended_order_qty"] == 210.0
