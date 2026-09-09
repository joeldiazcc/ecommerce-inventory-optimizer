"""Tests del coste de la política: margen perdido vs capital inmovilizado."""

import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.features import clean_transactions
from inventario_ecommerce.modeling.economics import (
    attach_costs,
    cost_by_policy,
    cost_sensitivity,
    optimal_z,
    optimal_z_by_class,
    unit_prices_by_sku,
    z_grid_policies,
)


def _by_sku(units_lost: list[float], avg_on_hand: list[float], policies: list[str]) -> pd.DataFrame:
    n = len(policies)
    return pd.DataFrame(
        {
            config.COL_STOCK_CODE: ["A1"] * n,
            config.COL_DESCRIPTION: ["TOP"] * n,
            "ABCClass": ["A"] * n,
            "policy": policies,
            "units_lost": units_lost,
            "avg_on_hand": avg_on_hand,
            "demand_total": [100.0] * n,
            "units_served": [100.0 - u for u in units_lost],
        }
    )


def _prices(price: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame({config.COL_STOCK_CODE: ["A1"], "unit_price": [price]})


def test_attach_costs_values_lost_margin_and_idle_capital():
    by_sku = _by_sku([10.0], [100.0], ["z_1.00"])

    costed = attach_costs(
        by_sku, _prices(10.0), margin_rate=0.4, annual_holding_rate=0.25, horizon_days=365
    )

    # 10 ud perdidas x 10 de precio x 40% de margen
    assert costed.loc[0, "stockout_cost"] == 40.0
    # 100 ud paradas x 6 de coste unitario x 25% durante un año
    assert costed.loc[0, "holding_cost"] == 150.0
    assert costed.loc[0, "total_cost"] == 190.0


def test_holding_cost_scales_with_the_simulated_window():
    by_sku = _by_sku([0.0], [100.0], ["z_1.00"])

    anual = attach_costs(by_sku, _prices(), 0.4, 0.25, horizon_days=365)
    mensual = attach_costs(by_sku, _prices(), 0.4, 0.25, horizon_days=30)

    assert mensual.loc[0, "holding_cost"] == anual.loc[0, "holding_cost"] * (30 / 365)


def test_sku_without_price_is_left_out_of_the_aggregate():
    by_sku = _by_sku([10.0, 5.0], [50.0, 80.0], ["z_0.00", "z_1.00"])
    by_sku.loc[1, config.COL_STOCK_CODE] = "SIN_PRECIO"

    costs = cost_by_policy(attach_costs(by_sku, _prices()))

    assert list(costs["policy"]) == ["z_0.00"]


def test_optimal_z_picks_the_cheapest_policy():
    by_sku = _by_sku([40.0, 10.0, 0.0], [20.0, 60.0, 400.0], ["z_0.00", "z_1.00", "z_3.00"])
    z_by_policy = {"z_0.00": 0.0, "z_1.00": 1.0, "z_3.00": 3.0}

    costed = attach_costs(by_sku, _prices(), 0.4, 0.25, horizon_days=365)
    best = optimal_z(cost_by_policy(costed), z_by_policy)

    # z=0: 160 + 30 = 190 | z=1: 40 + 90 = 130 | z=3: 0 + 600 = 600
    assert best["z"] == 1.0
    assert best["total_cost"] == 130.0


def test_optimal_z_by_class_decides_per_class():
    frames = []
    for sku, klass, lost, on_hand in (
        # A: 40 ud perdidas a 100 € son caras; 80 ud paradas a coste 60 aún salen baratas.
        ("A1", "A", [40.0, 0.0], [20.0, 80.0]),
        # C: el SKU barato no justifica mil unidades paradas.
        ("C1", "C", [40.0, 0.0], [20.0, 1000.0]),
    ):
        frame = _by_sku(lost, on_hand, ["z_0.00", "z_3.00"])
        frame[config.COL_STOCK_CODE] = sku
        frame["ABCClass"] = klass
        frames.append(frame)
    by_sku = pd.concat(frames, ignore_index=True)
    prices = pd.DataFrame({config.COL_STOCK_CODE: ["A1", "C1"], "unit_price": [100.0, 1.0]})

    costed = attach_costs(by_sku, prices, 0.4, 0.25, horizon_days=365)
    best = optimal_z_by_class(costed, {"z_0.00": 0.0, "z_3.00": 3.0}).set_index("ABCClass")

    # El SKU caro compensa cubrirse; el barato no paga tanto stock parado.
    assert best.loc["A", "z"] == 3.0
    assert best.loc["C", "z"] == 0.0


def test_more_margin_pushes_the_optimum_towards_more_service():
    by_sku = _by_sku([40.0, 10.0, 0.0], [20.0, 60.0, 400.0], ["z_0.00", "z_1.00", "z_3.00"])
    z_by_policy = {"z_0.00": 0.0, "z_1.00": 1.0, "z_3.00": 3.0}

    sensitivity = cost_sensitivity(
        by_sku,
        _prices(),
        z_by_policy,
        margin_rates=(0.2, 0.8),
        holding_rates=(0.25,),
        horizon_days=365,
    )

    barato = sensitivity[sensitivity["margin_rate"] == 0.2]["z_optimo"].iloc[0]
    caro = sensitivity[sensitivity["margin_rate"] == 0.8]["z_optimo"].iloc[0]
    assert caro >= barato


def test_unit_prices_use_the_median_of_the_invoice_lines():
    raw = pd.DataFrame(
        {
            config.COL_INVOICE: ["1", "2", "3"],
            config.COL_STOCK_CODE: ["A1", "A1", "A1"],
            config.COL_DESCRIPTION: ["TOP"] * 3,
            config.COL_QUANTITY: [1, 1, 1],
            config.COL_INVOICE_DATE: ["2011-12-01", "2011-12-02", "2011-12-03"],
            config.COL_PRICE: [2.0, 3.0, 100.0],
        }
    )

    prices = unit_prices_by_sku(clean_transactions(raw))

    assert prices.loc[0, "unit_price"] == 3.0


def test_z_grid_policies_cover_the_configured_grid():
    policies = z_grid_policies((0.0, 1.5))

    assert [p.name for p in policies] == ["z_0.00", "z_1.50"]
    assert [p.z for p in policies] == [0.0, 1.5]
