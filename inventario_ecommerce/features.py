"""Limpieza, agregaciones y features para análisis de inventario."""

import numpy as np
import pandas as pd

from inventario_ecommerce import config


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia nulos, cancela devoluciones y tipa columnas clave."""
    cleaned = df.copy()

    # Normalizar nombres si vienen en formato Online Retail clásico
    rename_map = {
        "InvoiceNo": config.COL_INVOICE,
        "UnitPrice": config.COL_PRICE,
        "CustomerID": config.COL_CUSTOMER_ID,
    }
    cleaned = cleaned.rename(columns={k: v for k, v in rename_map.items() if k in cleaned.columns})

    required = [
        config.COL_STOCK_CODE,
        config.COL_DESCRIPTION,
        config.COL_QUANTITY,
        config.COL_INVOICE_DATE,
        config.COL_PRICE,
    ]
    missing = [c for c in required if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    cleaned[config.COL_INVOICE_DATE] = pd.to_datetime(
        cleaned[config.COL_INVOICE_DATE], errors="coerce"
    )

    # Nulos críticos: fecha, producto o precio
    cleaned = cleaned.dropna(
        subset=[
            config.COL_STOCK_CODE,
            config.COL_DESCRIPTION,
            config.COL_INVOICE_DATE,
            config.COL_PRICE,
        ]
    )

    cleaned[config.COL_DESCRIPTION] = cleaned[config.COL_DESCRIPTION].astype(str).str.strip()
    cleaned = cleaned[cleaned[config.COL_DESCRIPTION] != ""]

    # Filtrar devoluciones / cantidades o precios no válidos
    cleaned = cleaned[
        (cleaned[config.COL_QUANTITY] > 0) & (cleaned[config.COL_PRICE] >= 0)
    ]

    # Quitar líneas que no representan stock físico de inventario.
    stock_upper = cleaned[config.COL_STOCK_CODE].astype(str).str.upper().str.strip()
    desc_lower = cleaned[config.COL_DESCRIPTION].astype(str).str.lower()
    keyword_pattern = "|".join(config.NON_PRODUCT_DESC_KEYWORDS)
    non_product_mask = stock_upper.isin(config.NON_PRODUCT_STOCK_CODES) | (
        desc_lower.str.contains(keyword_pattern, regex=True, na=False)
    )
    # Descripción exacta "check" / "manual" (basura), sin matchear "CHECK hammock", etc.
    exact_junk_desc = desc_lower.isin({"check", "manual", "adjustment"})
    cleaned = cleaned[~(non_product_mask | exact_junk_desc)]

    cleaned["Sales"] = cleaned[config.COL_QUANTITY] * cleaned[config.COL_PRICE]
    return cleaned.reset_index(drop=True)


def sales_by_product_last_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventas totales por producto en el último trimestre disponible."""
    if df.empty:
        raise ValueError("El DataFrame está vacío tras la limpieza.")

    max_date = df[config.COL_INVOICE_DATE].max()
    start_quarter = (max_date - pd.DateOffset(months=3)) + pd.Timedelta(days=1)
    last_q = df[
        (df[config.COL_INVOICE_DATE] >= start_quarter)
        & (df[config.COL_INVOICE_DATE] <= max_date)
    ].copy()

    summary = (
        last_q.groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .agg(
            QuantitySold=(config.COL_QUANTITY, "sum"),
            TotalSales=("Sales", "sum"),
            NTransactions=(config.COL_INVOICE_DATE, "count"),
        )
        .sort_values("TotalSales", ascending=False)
        .reset_index(drop=True)
    )
    summary.attrs["period_start"] = start_quarter
    summary.attrs["period_end"] = max_date
    return summary


def build_daily_sku_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la demanda diaria por SKU."""
    daily = df.copy()
    daily["Date"] = daily[config.COL_INVOICE_DATE].dt.floor("D")
    result = (
        daily.groupby(
            ["Date", config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False
        ).agg(
            QuantitySold=(config.COL_QUANTITY, "sum"),
            DailySales=("Sales", "sum"),
            NTransactions=(config.COL_INVOICE, "nunique"),
        )
    )
    return result.sort_values(["Date", config.COL_STOCK_CODE]).reset_index(drop=True)


def fill_missing_demand_days(daily_sku_demand: pd.DataFrame) -> pd.DataFrame:
    """Rellena días sin venta con 0 desde la 1.ª venta de cada SKU hasta el max global.

    Sin esto, rolling/media solo ven días con transacción y sesgan la demanda al alza.
    """
    if daily_sku_demand.empty:
        return daily_sku_demand

    group_cols = [config.COL_STOCK_CODE, config.COL_DESCRIPTION]
    daily = daily_sku_demand.copy()
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
    global_end = daily["Date"].max()

    frames: list[pd.DataFrame] = []
    for (stock, desc), grp in daily.groupby(group_cols, sort=False):
        full_idx = pd.date_range(grp["Date"].min(), global_end, freq="D")
        aligned = grp.set_index("Date").reindex(full_idx)
        aligned[config.COL_STOCK_CODE] = stock
        aligned[config.COL_DESCRIPTION] = desc
        aligned["QuantitySold"] = aligned["QuantitySold"].fillna(0.0)
        aligned["DailySales"] = aligned["DailySales"].fillna(0.0)
        aligned["NTransactions"] = aligned["NTransactions"].fillna(0)
        frames.append(aligned)

    out = pd.concat(frames)
    out.index.name = "Date"
    return (
        out.reset_index()
        .sort_values(["Date", config.COL_STOCK_CODE])
        .reset_index(drop=True)
    )


def winsorize_daily_quantity(
    daily_sku_demand: pd.DataFrame,
    percentile: float | None = None,
) -> pd.DataFrame:
    """Cap QuantitySold diaria al percentil global (días con venta > 0)."""
    if daily_sku_demand.empty:
        return daily_sku_demand

    pct = (
        config.DAILY_QTY_WINSOR_PERCENTILE if percentile is None else percentile
    )
    out = daily_sku_demand.copy()
    positive = out.loc[out["QuantitySold"] > 0, "QuantitySold"]
    if positive.empty:
        return out

    cap = float(positive.quantile(pct))
    out["QuantitySold"] = out["QuantitySold"].clip(upper=cap)
    return out


def drop_outlier_skus(daily_sku_demand: pd.DataFrame) -> pd.DataFrame:
    """Excluye SKUs one-shot conocidos del modelado de demanda."""
    if daily_sku_demand.empty:
        return daily_sku_demand
    stock = daily_sku_demand[config.COL_STOCK_CODE].astype(str).str.strip()
    return daily_sku_demand[
        ~stock.isin(config.OUTLIER_STOCK_CODES)
    ].reset_index(drop=True)


def prepare_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline diario: agregación → ceros → winsor → drop outliers."""
    daily = build_daily_sku_demand(df)
    daily = fill_missing_demand_days(daily)
    daily = winsorize_daily_quantity(daily)
    daily = drop_outlier_skus(daily)
    return daily


def compute_abc_classification(
    sales_summary: pd.DataFrame,
    sales_col: str = "TotalSales",
) -> pd.DataFrame:
    """Clasifica SKUs por contribución acumulada de ventas (A/B/C)."""
    abc = sales_summary.copy()
    if abc.empty:
        return abc

    abc = abc.sort_values(sales_col, ascending=False).reset_index(drop=True)
    total_sales = abc[sales_col].sum()
    abc["SalesShare"] = np.where(total_sales > 0, abc[sales_col] / total_sales, 0.0)
    abc["CumSalesShare"] = abc["SalesShare"].cumsum()

    abc["ABCClass"] = np.select(
        [
            abc["CumSalesShare"] <= 0.80,
            abc["CumSalesShare"] <= 0.95,
        ],
        ["A", "B"],
        default="C",
    )
    return abc


def build_rolling_features(
    daily_sku_demand: pd.DataFrame,
    windows: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    """Crea features rolling de demanda por SKU."""
    features = daily_sku_demand.copy()
    features = features.sort_values(["Date", config.COL_STOCK_CODE]).reset_index(drop=True)
    group_cols = [config.COL_STOCK_CODE, config.COL_DESCRIPTION]

    for window in windows:
        grouped = features.groupby(group_cols, sort=False)["QuantitySold"]
        features[f"demand_mean_{window}d"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        features[f"demand_std_{window}d"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).std()
        ).fillna(0.0)
        features[f"demand_sum_{window}d"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).sum()
        )

    features["demand_cv_30d"] = np.where(
        features["demand_mean_30d"] > 0,
        features["demand_std_30d"] / features["demand_mean_30d"],
        0.0,
    )
    return features
