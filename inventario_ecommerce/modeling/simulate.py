"""Simulación de la política de reposición sobre el holdout.

El backtest de forecast mide el error de la predicción. Esto mide la decisión:
cuántas unidades se dejan de servir y cuánto stock hay que sostener para lograrlo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.dataset import load_transactions, save_processed
from inventario_ecommerce.features import (
    clean_transactions,
    compute_abc_classification,
    prepare_daily_demand,
    sales_by_product_last_quarter,
)
from inventario_ecommerce.modeling.ets import forecast_ets_class_a
from inventario_ecommerce.plots import plot_policy_service_tradeoff


@dataclass(frozen=True)
class PolicySpec:
    """Una política = cómo se fija el stock de seguridad y con qué forecast."""

    name: str
    label: str
    z: float | None = None  # None → z por clase ABC
    forecast_model: str = "ma30"


DEFAULT_POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec("media", "Pedir la media (sin stock de seguridad)", z=0.0),
    PolicySpec("z_clase", "ROP con z por clase ABC", z=None),
    PolicySpec("z_90", "ROP con z = 1.28 (90%)", z=1.28),
    PolicySpec("z_95", "ROP con z = 1.65 (95%)", z=1.65),
    PolicySpec("z_99", "ROP con z = 2.33 (99%)", z=2.33),
)

ETS_POLICY = PolicySpec(
    "z_clase_ets", "ROP con z por clase ABC + forecast ETS", z=None, forecast_model="ets"
)


def simulate_sku(
    demand: np.ndarray,
    reorder_point: float,
    target_stock: float,
    lead_time_days: int = config.LEAD_TIME_DAYS,
    review_period_days: int = config.REVIEW_PERIOD_DAYS,
    initial_on_hand: float | None = None,
) -> dict[str, float]:
    """Revisión periódica con venta perdida.

    Cada día: llegan los pedidos pendientes → se sirve la demanda con lo que hay →
    en día de revisión, si la posición de inventario cayó al punto de reorden se
    pide hasta el objetivo. Lo que no se sirve se pierde, no queda en backorder.
    """
    demand = np.asarray(demand, dtype=float)
    n_days = len(demand)
    if n_days == 0:
        raise ValueError("La serie de demanda del holdout está vacía.")

    on_hand = float(target_stock if initial_on_hand is None else initial_on_hand)
    on_order = 0.0
    arrivals = np.zeros(n_days + lead_time_days + 1)

    units_served = 0.0
    units_lost = 0.0
    stockout_days = 0
    on_hand_sum = 0.0
    orders_placed = 0
    units_ordered = 0.0

    for day in range(n_days):
        received = arrivals[day]
        on_hand += received
        on_order -= received

        served = min(on_hand, demand[day])
        on_hand -= served
        units_served += served
        unmet = demand[day] - served
        if unmet > 1e-9:
            units_lost += unmet
            stockout_days += 1

        if day % review_period_days == 0:
            position = on_hand + on_order
            if position <= reorder_point:
                qty = max(target_stock - position, 0.0)
                if qty > 0:
                    arrivals[day + lead_time_days] += qty
                    on_order += qty
                    orders_placed += 1
                    units_ordered += qty

        on_hand_sum += on_hand

    demand_total = float(demand.sum())
    avg_on_hand = on_hand_sum / n_days
    mean_daily_demand = demand_total / n_days
    return {
        "demand_total": demand_total,
        "units_served": units_served,
        "units_lost": units_lost,
        "fill_rate": units_served / demand_total if demand_total > 0 else 1.0,
        "stockout_days": stockout_days,
        "avg_on_hand": avg_on_hand,
        "days_of_cover": (
            avg_on_hand / mean_daily_demand if mean_daily_demand > 0 else np.nan
        ),
        "orders_placed": orders_placed,
        "units_ordered": units_ordered,
        "ending_on_hand": on_hand,
    }


def build_simulation_inputs(
    daily_sku_demand: pd.DataFrame,
    abc: pd.DataFrame,
    horizon_days: int = 30,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Una fila por SKU: estadísticos de la ventana de entrenamiento y demanda real del holdout.

    Todo lo que alimenta la política sale de datos anteriores al corte; el holdout
    solo se usa para consumir stock.
    """
    daily = daily_sku_demand.copy()
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
    cutoff = daily["Date"].max() - pd.Timedelta(days=horizon_days)

    train = daily[daily["Date"] <= cutoff]
    test = daily[daily["Date"] > cutoff]
    if train.empty or test.empty:
        raise ValueError("No hay suficientes datos para simular la política.")

    group_cols = [config.COL_STOCK_CODE, config.COL_DESCRIPTION]
    tail = train[train["Date"] > (cutoff - pd.Timedelta(days=lookback_days))]
    stats = (
        tail.groupby(group_cols, as_index=False)
        .agg(
            demand_mean_30d=("QuantitySold", "mean"),
            demand_std_30d=("QuantitySold", "std"),
            lookback_qty=("QuantitySold", "sum"),
        )
        .fillna({"demand_std_30d": 0.0})
    )
    stats = stats[stats["lookback_qty"] > 0].drop(columns=["lookback_qty"])

    holdout = (
        test.sort_values("Date")
        .groupby(group_cols)["QuantitySold"]
        .apply(lambda s: np.asarray(s, dtype=float))
        .rename("holdout_demand")
        .reset_index()
    )

    inputs = stats.merge(holdout, on=group_cols, how="inner").merge(
        abc[[config.COL_STOCK_CODE, config.COL_DESCRIPTION, "ABCClass"]],
        on=group_cols,
        how="inner",
    )
    inputs.attrs["cutoff"] = cutoff
    inputs.attrs["horizon_days"] = horizon_days
    return inputs.reset_index(drop=True)


def _forecast_for_policy(
    inputs: pd.DataFrame,
    policy: PolicySpec,
    ets_forecast: pd.DataFrame | None,
) -> pd.Series:
    """Demanda diaria prevista: media móvil, o ETS donde haya ajuste."""
    forecast = inputs["demand_mean_30d"].astype(float)
    if policy.forecast_model != "ets" or ets_forecast is None or ets_forecast.empty:
        return forecast

    ets = ets_forecast.set_index(
        [config.COL_STOCK_CODE, config.COL_DESCRIPTION]
    )["forecast_daily"]
    keys = pd.MultiIndex.from_frame(
        inputs[[config.COL_STOCK_CODE, config.COL_DESCRIPTION]]
    )
    return pd.Series(keys.map(ets), index=inputs.index).astype(float).fillna(forecast)


def simulate_policies(
    daily_sku_demand: pd.DataFrame,
    abc: pd.DataFrame,
    policies: tuple[PolicySpec, ...] = DEFAULT_POLICIES,
    horizon_days: int = 30,
    lookback_days: int = 30,
    lead_time_days: int = config.LEAD_TIME_DAYS,
    review_period_days: int = config.REVIEW_PERIOD_DAYS,
    ets_forecast: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula cada política sobre el mismo holdout y devuelve detalle por SKU y resumen."""
    inputs = build_simulation_inputs(daily_sku_demand, abc, horizon_days, lookback_days)
    if inputs.empty:
        raise ValueError("Ningún SKU cumple los requisitos para simular.")

    sqrt_lt = np.sqrt(lead_time_days)
    records: list[dict] = []

    for policy in policies:
        forecast_daily = _forecast_for_policy(inputs, policy, ets_forecast)
        if policy.z is None:
            z = (
                inputs["ABCClass"]
                .map(config.SERVICE_Z_BY_CLASS)
                .fillna(config.DEFAULT_SERVICE_Z)
                .astype(float)
            )
        else:
            z = pd.Series(float(policy.z), index=inputs.index)

        safety_stock = z * inputs["demand_std_30d"].fillna(0.0) * sqrt_lt
        reorder_point = forecast_daily * lead_time_days + safety_stock
        target_stock = reorder_point + forecast_daily * review_period_days

        for i, row in enumerate(inputs.itertuples(index=False)):
            result = simulate_sku(
                row.holdout_demand,
                reorder_point=float(reorder_point.iat[i]),
                target_stock=float(target_stock.iat[i]),
                lead_time_days=lead_time_days,
                review_period_days=review_period_days,
            )
            records.append(
                {
                    config.COL_STOCK_CODE: getattr(row, config.COL_STOCK_CODE),
                    config.COL_DESCRIPTION: getattr(row, config.COL_DESCRIPTION),
                    "ABCClass": row.ABCClass,
                    "policy": policy.name,
                    "forecast_model": policy.forecast_model,
                    "forecast_daily": float(forecast_daily.iat[i]),
                    "z_service": float(z.iat[i]),
                    "safety_stock": float(safety_stock.iat[i]),
                    "reorder_point": float(reorder_point.iat[i]),
                    "target_stock": float(target_stock.iat[i]),
                    **result,
                }
            )
        print(f"Simulada política {policy.name} sobre {len(inputs):,} SKUs", flush=True)

    by_sku = pd.DataFrame(records)
    summary = summarize_simulation(by_sku, policies, horizon_days)
    summary.insert(0, "CutoffDate", inputs.attrs["cutoff"].date().isoformat())
    return by_sku, summary


def summarize_simulation(
    by_sku: pd.DataFrame,
    policies: tuple[PolicySpec, ...] = DEFAULT_POLICIES,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Agrega el detalle por SKU a una fila por política."""
    labels = {p.name: p.label for p in policies}
    rows: list[dict] = []

    for name, grp in by_sku.groupby("policy", sort=False):
        demand_total = float(grp["demand_total"].sum())
        units_served = float(grp["units_served"].sum())
        avg_on_hand = float(grp["avg_on_hand"].sum())
        sku_days = len(grp) * horizon_days
        rows.append(
            {
                "policy": name,
                "label": labels.get(name, name),
                "SKUs": int(len(grp)),
                "demand_total": demand_total,
                "units_served": units_served,
                "units_lost": float(grp["units_lost"].sum()),
                "fill_rate": units_served / demand_total if demand_total > 0 else np.nan,
                "skus_with_stockout": int((grp["stockout_days"] > 0).sum()),
                "stockout_day_share": float(grp["stockout_days"].sum()) / sku_days,
                "avg_on_hand_units": avg_on_hand,
                "days_of_cover": (
                    avg_on_hand / (demand_total / horizon_days)
                    if demand_total > 0
                    else np.nan
                ),
                "orders_placed": int(grp["orders_placed"].sum()),
                "units_ordered": float(grp["units_ordered"].sum()),
            }
        )

    order = {p.name: i for i, p in enumerate(policies)}
    summary = pd.DataFrame(rows)
    return (
        summary.sort_values("policy", key=lambda s: s.map(order).fillna(len(order)))
        .reset_index(drop=True)
    )


def fill_rate_by_class(by_sku: pd.DataFrame) -> pd.DataFrame:
    """Fill rate y stock medio por política y clase ABC."""
    grouped = by_sku.groupby(["policy", "ABCClass"], as_index=False).agg(
        SKUs=(config.COL_STOCK_CODE, "count"),
        demand_total=("demand_total", "sum"),
        units_served=("units_served", "sum"),
        units_lost=("units_lost", "sum"),
        avg_on_hand_units=("avg_on_hand", "sum"),
    )
    grouped["fill_rate"] = np.where(
        grouped["demand_total"] > 0,
        grouped["units_served"] / grouped["demand_total"],
        np.nan,
    )
    return grouped


def simulate(with_ets: bool = True) -> dict[str, pd.DataFrame]:
    """Pipeline de simulación completo; guarda detalle, resumen y figura."""
    raw = load_transactions()
    clean = clean_transactions(raw)
    abc = compute_abc_classification(sales_by_product_last_quarter(clean))
    daily = prepare_daily_demand(clean, abc=abc)

    policies = DEFAULT_POLICIES
    ets_forecast = None
    if with_ets:
        cutoff = pd.to_datetime(daily["Date"]).max() - pd.Timedelta(days=30)
        train = daily[pd.to_datetime(daily["Date"]) <= cutoff]
        ets_forecast = forecast_ets_class_a(train, abc, horizon_days=30)
        if not ets_forecast.empty:
            policies = (*DEFAULT_POLICIES, ETS_POLICY)

    by_sku, summary = simulate_policies(daily, abc, policies=policies, ets_forecast=ets_forecast)
    by_class = fill_rate_by_class(by_sku)

    save_processed(by_sku, "policy_simulation_by_sku.csv")
    save_processed(summary, "policy_simulation_summary.csv")
    save_processed(by_class, "policy_simulation_by_class.csv")
    plot_policy_service_tradeoff(summary, save=True)

    return {"by_sku": by_sku, "summary": summary, "by_class": by_class}


if __name__ == "__main__":
    outputs = simulate()
    cols = ["policy", "fill_rate", "units_lost", "avg_on_hand_units", "days_of_cover"]
    print("\nSimulación de política completada.")
    print(outputs["summary"][cols].to_string(index=False))
    print(f"\nSalida: {config.PROCESSED_DATA_DIR}")
