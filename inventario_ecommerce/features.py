"""Limpieza y features básicas para análisis de inventario."""

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
