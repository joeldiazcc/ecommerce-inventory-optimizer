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


def save_processed(df: pd.DataFrame, filename: str = "sales_last_quarter_by_product.csv") -> Path:
    """Persiste un dataset procesado en data/processed/."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
