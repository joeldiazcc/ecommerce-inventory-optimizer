"""Carga y generación de datasets transaccionales de retail."""

from pathlib import Path

import pandas as pd

from inventario_ecommerce import config


def load_transactions(path: Path | None = None) -> pd.DataFrame:
    """Carga un CSV/Excel de transacciones retail desde data/raw/."""
    file_path = Path(path) if path else config.DEFAULT_RAW_FILE
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {file_path}. "
            "Coloca Online Retail II (UCI) u Olist en data/raw/."
        )

    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    return pd.read_csv(file_path, encoding="utf-8", low_memory=False)


def load_stock_positions(path: Path | None = None) -> pd.DataFrame | None:
    """Carga stock físico y pedidos en curso por SKU, si existe la fuente.

    Espera `StockCode` y al menos una de `on_hand` / `on_order`. Devuelve None
    cuando no hay fichero: el dataset de Online Retail II son ventas, no inventario.
    """
    file_path = Path(path) if path else config.STOCK_ON_HAND_FILE
    if not file_path.exists():
        return None

    positions = pd.read_csv(file_path)
    if config.COL_STOCK_CODE not in positions.columns:
        raise ValueError(
            f"{file_path} debe tener la columna {config.COL_STOCK_CODE}."
        )
    if not {"on_hand", "on_order"} & set(positions.columns):
        raise ValueError(f"{file_path} debe tener on_hand y/o on_order.")

    positions[config.COL_STOCK_CODE] = (
        positions[config.COL_STOCK_CODE].astype(str).str.strip()
    )
    for col in ("on_hand", "on_order"):
        positions[col] = (
            pd.to_numeric(positions.get(col), errors="coerce").fillna(0.0)
            if col in positions.columns
            else 0.0
        )
    return positions.groupby(config.COL_STOCK_CODE, as_index=False)[
        ["on_hand", "on_order"]
    ].sum()


def save_processed(df: pd.DataFrame, filename: str = "sales_last_quarter_by_product.csv") -> Path:
    """Persiste un dataset procesado en data/processed/."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
