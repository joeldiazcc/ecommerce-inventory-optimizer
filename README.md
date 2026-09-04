# Optimización de Inventario E-commerce

Análisis de demanda retail a partir de transacciones ([Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)): limpieza del histórico y ventas por producto en el último trimestre, base para ABC, forecast y punto de reorden.

![Top 10 productos por ventas — último trimestre](reports/figures/top_products_last_quarter.png)

## Qué hay en este repo

| Pieza | Descripción |
|-------|-------------|
| `notebooks/01_carga_limpieza_retail.ipynb` | Fase 1: carga, limpieza y ventas del último trimestre |
| `notebooks/02_forecast_reorder_baseline.ipynb` | Fase 2: ABC, forecast baseline y punto de reorden |
| `notebooks/03_demand_hygiene.ipynb` | Fase 3: ceros en calendario, winsor y filtro no-producto |
| `inventario_ecommerce/` | Código reutilizable (`dataset`, `features`, `modeling`, `plots`) |
| `references/data_dictionary.md` | Columnas y reglas de limpieza |
| `data/raw/sample_online_retail.csv` | Sample para smoke-test sin Kaggle |

## Resultados (Online Retail II)

| Métrica | Valor |
|--------|------:|
| Filas raw | ~1.07M |
| Periodo (último trimestre) | 2011-09-10 → 2011-12-09 |
| SKUs ABC (último trimestre) | 3,393 |
| Recomendaciones (activos lookback ∩ ABC) | 2,963 |
| MAE baseline (backtest 30d, con días a 0) | 4.06 ud/día |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_carga_limpieza_retail.ipynb
```

Para generar artefactos de modelado desde terminal:

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
references/               # diccionario de datos
reports/figures/          # gráfico de ejemplo versionado
```

## Entregables generados (local)

- `data/processed/abc_last_quarter.csv`
- `data/processed/sku_rolling_features_latest.csv`
- `data/processed/forecast_backtest_by_sku.csv`
- `data/processed/forecast_backtest_global.csv`
- `data/processed/inventory_reorder_recommendations.csv`

## Próximas mejoras

1. Forecast por familia / top-N clase A (Prophet, ETS o LightGBM) vs baseline  
2. Incorporar stock actual y lead time real por proveedor  
3. Simulador de quiebre/sobrestock por política  

## Stack

Python · pandas · numpy · matplotlib · Jupyter
