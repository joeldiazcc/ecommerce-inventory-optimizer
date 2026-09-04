# Optimización de Inventario E-commerce

De transacciones retail a una **tabla de reorden por SKU**: limpieza del histórico, demanda diaria (con días sin venta), clasificación ABC, forecast baseline y punto de reorden.

Fuente: [Online Retail II (UCI / Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).

![Top 10 productos por ventas — último trimestre](reports/figures/top_products_last_quarter.png)

## Pipeline

1. **Limpieza** — nulos, devoluciones y líneas que no son producto físico
2. **Demanda diaria** — calendario continuo (ceros), cap P99 y exclusión de one-shots
3. **ABC** — 80% / 95% de ventas acumuladas del último trimestre
4. **Forecast** — media diaria de los últimos 30 días, backtest temporal (MAE / MAPE)
5. **Reorden** — lead time 14 d, revisión 7 d, z según clase A/B/C

El dataset no trae stock on-hand: `recommended_order_qty` es la demanda del ciclo de revisión (`forecast_daily × 7`).

## Resultados (Online Retail II)

| Métrica | Valor |
|--------|------:|
| Filas raw | ~1.07M |
| Periodo (último trimestre) | 2011-09-10 → 2011-12-09 |
| SKUs ABC (último trimestre) | 3,393 |
| Recomendaciones (activos lookback ∩ ABC) | 2,963 |
| MAE baseline (backtest 30d, calendario con ceros) | 4.06 ud/día |

El MAE se evalúa sobre **todos** los días del holdout, no solo los días con ticket. Es la métrica adecuada para demanda intermitente; no es comparable a un MAE calculado solo en días con venta.

## Qué hay en este repo

| Pieza | Descripción |
|-------|-------------|
| `notebooks/01_carga_limpieza_retail.ipynb` | Carga, calidad, limpieza y ventas del último trimestre |
| `notebooks/02_forecast_reorder_baseline.ipynb` | ABC, forecast, backtest y tabla de reorden |
| `notebooks/03_demand_hygiene.ipynb` | Por qué se rellenan ceros y se recortan outliers |
| `inventario_ecommerce/` | Código reutilizable (`dataset`, `features`, `modeling`, `plots`) |
| `data/raw/sample_online_retail.csv` | Sample para smoke-test sin Kaggle |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_carga_limpieza_retail.ipynb
```

Notebooks en orden: **01** (datos) → **02** (modelo y reorden). La **03** explica la higiene de demanda; no sustituye a la 02.

Pipeline completo desde terminal:

```bash
python -m inventario_ecommerce.modeling.train
python -m inventario_ecommerce.modeling.predict
```

### Dataset completo

El CSV grande (~90 MB) **no** está en GitHub.

1. Descarga [Online Retail II UCI (Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
2. Guárdalo como `data/raw/online_retail_II.csv`

Sin él, el sample local basta para validar el código.

## Estructura

```
data/raw|interim|processed|external
notebooks/
inventario_ecommerce/     # config, carga, features, modeling, plots
reports/figures/          # gráfico de ejemplo versionado
```

## Artefactos generados (local)

- `data/processed/sales_last_quarter_by_product.csv`
- `data/processed/abc_last_quarter.csv`
- `data/processed/sku_rolling_features_latest.csv`
- `data/processed/forecast_backtest_by_sku.csv`
- `data/processed/forecast_backtest_global.csv`
- `data/processed/inventory_reorder_recommendations.csv`

## Stack

Python · pandas · numpy · matplotlib · Jupyter
