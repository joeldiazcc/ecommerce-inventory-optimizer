"""Coste de la política: margen perdido frente a capital inmovilizado.

La simulación cuenta unidades. Aquí se les pone precio, porque una unidad no
servida de un SKU caro y una unidad parada de uno barato no pesan igual. El
dataset trae precio de venta pero no coste de compra: el margen y la tasa de
posesión son supuestos declarados, no datos.
"""

from __future__ import annotations

import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.dataset import load_transactions, save_processed
from inventario_ecommerce.features import (
    clean_transactions,
    compute_abc_classification,
    prepare_daily_demand,
    sales_by_product_last_quarter,
)
from inventario_ecommerce.modeling.simulate import PolicySpec, simulate_policies
from inventario_ecommerce.plots import plot_cost_vs_service


def unit_prices_by_sku(clean: pd.DataFrame, last_quarter_only: bool = True) -> pd.DataFrame:
    """Precio unitario por SKU: mediana de las líneas de factura.

    Mediana y no media: el mismo SKU se vende a precios distintos según volumen
    y canal, y algunas líneas llevan precios promocionales extremos.
    """
    prices = clean.copy()
    if last_quarter_only:
        max_date = prices[config.COL_INVOICE_DATE].max()
        start = (max_date - pd.DateOffset(months=3)) + pd.Timedelta(days=1)
        prices = prices[prices[config.COL_INVOICE_DATE] >= start]

    prices = prices[prices[config.COL_PRICE] > 0]
    out = (
        prices.groupby(config.COL_STOCK_CODE, as_index=False)
        .agg(unit_price=(config.COL_PRICE, "median"))
    )
    out[config.COL_STOCK_CODE] = out[config.COL_STOCK_CODE].astype(str).str.strip()
    return out


def attach_costs(
    by_sku: pd.DataFrame,
    prices: pd.DataFrame,
    margin_rate: float | None = None,
    annual_holding_rate: float | None = None,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Valora en dinero cada fila de la simulación.

    - Quiebre: margen bruto que se deja de ingresar por cada unidad no servida.
    - Posesión: coste de tener el stock medio parado durante el horizonte.
    """
    margin = config.GROSS_MARGIN_RATE if margin_rate is None else margin_rate
    holding = (
        config.ANNUAL_HOLDING_RATE if annual_holding_rate is None else annual_holding_rate
    )

    out = by_sku.copy()
    out["_stock"] = out[config.COL_STOCK_CODE].astype(str).str.strip()
    out = out.merge(
        prices.rename(columns={config.COL_STOCK_CODE: "_stock"}),
        on="_stock",
        how="left",
    ).drop(columns=["_stock"])

    # Un SKU sin precio no puede valorarse; se queda fuera del agregado.
    out["unit_price"] = out["unit_price"].astype(float)
    out["unit_cost"] = out["unit_price"] * (1.0 - margin)

    out["stockout_cost"] = out["units_lost"] * out["unit_price"] * margin
    out["holding_cost"] = (
        out["avg_on_hand"] * out["unit_cost"] * holding * (horizon_days / 365.0)
    )
    out["total_cost"] = out["stockout_cost"] + out["holding_cost"]
    return out


def cost_by_policy(by_sku_costed: pd.DataFrame) -> pd.DataFrame:
    """Coste agregado por política, sobre los SKUs con precio conocido."""
    valued = by_sku_costed.dropna(subset=["unit_price"])
    summary = valued.groupby("policy", as_index=False).agg(
        SKUs=(config.COL_STOCK_CODE, "count"),
        units_lost=("units_lost", "sum"),
        avg_on_hand_units=("avg_on_hand", "sum"),
        stockout_cost=("stockout_cost", "sum"),
        holding_cost=("holding_cost", "sum"),
        total_cost=("total_cost", "sum"),
        demand_total=("demand_total", "sum"),
        units_served=("units_served", "sum"),
    )
    summary["fill_rate"] = summary["units_served"] / summary["demand_total"]
    return summary


def z_grid_policies(z_grid: tuple[float, ...] | None = None) -> tuple[PolicySpec, ...]:
    """Una política por cada z del barrido."""
    grid = config.SERVICE_Z_GRID if z_grid is None else z_grid
    return tuple(
        PolicySpec(f"z_{z:.2f}", f"ROP con z = {z:.2f}", z=float(z)) for z in grid
    )


def _minimum_row(frame: pd.DataFrame, cost_col: str = "total_cost") -> pd.Series:
    return frame.loc[frame[cost_col].idxmin()]


def optimal_z(costs: pd.DataFrame, z_by_policy: dict[str, float]) -> dict[str, float]:
    """z de coste mínimo y cuánto ahorra frente a la política actual."""
    frame = costs.assign(z=costs["policy"].map(z_by_policy))
    best = _minimum_row(frame)
    return {
        "z": float(best["z"]),
        "total_cost": float(best["total_cost"]),
        "fill_rate": float(best["fill_rate"]),
    }


def optimal_z_by_class(
    by_sku_costed: pd.DataFrame,
    z_by_policy: dict[str, float],
) -> pd.DataFrame:
    """z de coste mínimo dentro de cada clase ABC.

    Cada clase puede permitirse un nivel de servicio distinto: lo que decide no
    es el volumen sino la relación entre el margen que se pierde y lo que cuesta
    tener esa unidad parada.
    """
    valued = by_sku_costed.dropna(subset=["unit_price"])
    grouped = valued.groupby(["ABCClass", "policy"], as_index=False).agg(
        stockout_cost=("stockout_cost", "sum"),
        holding_cost=("holding_cost", "sum"),
        total_cost=("total_cost", "sum"),
        demand_total=("demand_total", "sum"),
        units_served=("units_served", "sum"),
    )
    grouped["fill_rate"] = grouped["units_served"] / grouped["demand_total"]
    grouped["z"] = grouped["policy"].map(z_by_policy)

    rows = [
        _minimum_row(grp)[["ABCClass", "z", "stockout_cost", "holding_cost", "total_cost", "fill_rate"]]
        for _, grp in grouped.groupby("ABCClass")
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def cost_sensitivity(
    by_sku: pd.DataFrame,
    prices: pd.DataFrame,
    z_by_policy: dict[str, float],
    margin_rates: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60),
    holding_rates: tuple[float, ...] = (0.15, 0.25, 0.50, 1.00),
    horizon_days: int = 30,
) -> pd.DataFrame:
    """z óptimo bajo distintos supuestos de margen y tasa de posesión.

    La simulación no depende de los costes, así que se revalora la misma tabla
    en vez de volver a simular.
    """
    rows: list[dict] = []
    for margin in margin_rates:
        for holding in holding_rates:
            costed = attach_costs(by_sku, prices, margin, holding, horizon_days)
            best = optimal_z(cost_by_policy(costed), z_by_policy)
            rows.append(
                {
                    "margin_rate": margin,
                    "annual_holding_rate": holding,
                    "z_optimo": best["z"],
                    "fill_rate": best["fill_rate"],
                    "total_cost": best["total_cost"],
                }
            )
    return pd.DataFrame(rows)


def evaluate_costs(
    z_grid: tuple[float, ...] | None = None,
    horizon_days: int = 30,
) -> dict[str, pd.DataFrame]:
    """Simula el barrido de z, lo valora en dinero y guarda los artefactos."""
    raw = load_transactions()
    clean = clean_transactions(raw)
    abc = compute_abc_classification(sales_by_product_last_quarter(clean))
    daily = prepare_daily_demand(clean, abc=abc)
    prices = unit_prices_by_sku(clean)

    policies = z_grid_policies(z_grid)
    z_by_policy = {p.name: float(p.z) for p in policies}
    by_sku, _ = simulate_policies(
        daily, abc, policies=policies, horizon_days=horizon_days
    )

    costed = attach_costs(by_sku, prices, horizon_days=horizon_days)
    costs = cost_by_policy(costed).assign(z=lambda d: d["policy"].map(z_by_policy))
    by_class = optimal_z_by_class(costed, z_by_policy)
    sensitivity = cost_sensitivity(by_sku, prices, z_by_policy, horizon_days=horizon_days)

    save_processed(costs, "policy_cost_by_z.csv")
    save_processed(by_class, "policy_cost_optimal_by_class.csv")
    save_processed(sensitivity, "policy_cost_sensitivity.csv")
    plot_cost_vs_service(costs, save=True)

    return {
        "costs": costs,
        "by_class": by_class,
        "sensitivity": sensitivity,
        "by_sku": costed,
    }


if __name__ == "__main__":
    outputs = evaluate_costs()
    costs = outputs["costs"]
    best = costs.loc[costs["total_cost"].idxmin()]
    print("\nCoste de la política por nivel de servicio:")
    print(
        costs[["z", "fill_rate", "stockout_cost", "holding_cost", "total_cost"]]
        .round(2)
        .to_string(index=False)
    )
    print(f"\nz de coste mínimo: {best['z']:.2f} (fill rate {best['fill_rate']:.1%})")
    print("\nÓptimo por clase ABC:")
    print(outputs["by_class"].round(2).to_string(index=False))
    print(f"\nSalida: {config.PROCESSED_DATA_DIR}")
