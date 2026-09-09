"""Tests del simulador de política de reposición."""

import numpy as np
import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.modeling.simulate import (
    PolicySpec,
    build_simulation_inputs,
    simulate_policies,
    simulate_sku,
    summarize_simulation,
)


def test_simulate_sku_serves_everything_with_ample_stock():
    result = simulate_sku(
        np.full(10, 10.0),
        reorder_point=500.0,
        target_stock=1000.0,
        lead_time_days=14,
        review_period_days=7,
    )

    assert result["units_served"] == 100.0
    assert result["units_lost"] == 0.0
    assert result["fill_rate"] == 1.0
    assert result["stockout_days"] == 0
    assert result["orders_placed"] == 0
    assert result["avg_on_hand"] == 945.0


def test_simulate_sku_loses_sales_when_replenishment_arrives_too_late():
    result = simulate_sku(
        np.full(10, 10.0),
        reorder_point=20.0,
        target_stock=25.0,
        lead_time_days=14,
        review_period_days=7,
    )

    # Empieza con 25 ud y el pedido tarda 14 días: nada llega dentro del holdout.
    assert result["units_served"] == 25.0
    assert result["units_lost"] == 75.0
    assert result["fill_rate"] == 0.25
    assert result["stockout_days"] == 8
    assert result["orders_placed"] == 2
    assert result["units_ordered"] == 25.0


def test_simulate_sku_receives_order_within_horizon():
    result = simulate_sku(
        np.full(10, 10.0),
        reorder_point=100.0,
        target_stock=150.0,
        lead_time_days=3,
        review_period_days=2,
    )

    assert result["units_lost"] == 0.0
    assert result["orders_placed"] >= 1
    # Sin recepciones el stock final sería 150 − 100; lo que sobra es lo que llegó.
    assert result["ending_on_hand"] > 50.0


def _synthetic_daily(days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2011-10-01", periods=days, freq="D")
    frames = []
    for stock, desc, mean in (("A1", "TOP", 40.0), ("C1", "TAIL", 3.0)):
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    config.COL_STOCK_CODE: stock,
                    config.COL_DESCRIPTION: desc,
                    "QuantitySold": rng.poisson(mean, days).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _synthetic_abc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.COL_STOCK_CODE: ["A1", "C1"],
            config.COL_DESCRIPTION: ["TOP", "TAIL"],
            "TotalSales": [5000.0, 100.0],
            "ABCClass": ["A", "C"],
        }
    )


def test_build_simulation_inputs_splits_train_and_holdout():
    inputs = build_simulation_inputs(
        _synthetic_daily(), _synthetic_abc(), horizon_days=10, lookback_days=20
    )

    assert len(inputs) == 2
    assert all(len(arr) == 10 for arr in inputs["holdout_demand"])
    assert set(inputs["ABCClass"]) == {"A", "C"}


def test_more_safety_stock_buys_service_with_more_inventory():
    policies = (
        PolicySpec("media", "Pedir la media", z=0.0),
        PolicySpec("z_99", "z = 2.33", z=2.33),
    )

    by_sku, summary = simulate_policies(
        _synthetic_daily(),
        _synthetic_abc(),
        policies=policies,
        horizon_days=10,
        lookback_days=20,
        lead_time_days=5,
        review_period_days=3,
    )

    media = summary.set_index("policy").loc["media"]
    z99 = summary.set_index("policy").loc["z_99"]

    assert len(by_sku) == 4
    assert z99["fill_rate"] >= media["fill_rate"]
    assert z99["avg_on_hand_units"] > media["avg_on_hand_units"]
    assert list(summary["policy"]) == ["media", "z_99"]


def test_summarize_simulation_aggregates_fill_rate_over_units():
    by_sku = pd.DataFrame(
        {
            config.COL_STOCK_CODE: ["A1", "C1"],
            config.COL_DESCRIPTION: ["TOP", "TAIL"],
            "ABCClass": ["A", "C"],
            "policy": ["media", "media"],
            "demand_total": [300.0, 100.0],
            "units_served": [270.0, 50.0],
            "units_lost": [30.0, 50.0],
            "stockout_days": [3, 0],
            "avg_on_hand": [10.0, 5.0],
            "orders_placed": [1, 2],
            "units_ordered": [100.0, 20.0],
        }
    )

    summary = summarize_simulation(by_sku, horizon_days=10)

    assert summary.loc[0, "fill_rate"] == 0.8
    assert summary.loc[0, "avg_on_hand_units"] == 15.0
    assert summary.loc[0, "skus_with_stockout"] == 1
    assert summary.loc[0, "stockout_day_share"] == 0.15
