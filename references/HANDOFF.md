# Estado del proyecto

Resumen técnico corto. Contexto de sesión y backlog: `references/RESUMEN_PROYECTO.md`.

| | |
|--|--|
| **Repo** | https://github.com/joeldiazcc/ecommerce-inventory-optimizer |
| **Local** | `c:\code\proyecto-a-inventario-ecommerce` |
| **Pipeline** | Limpieza → ABC → demanda diaria (ceros + winsor + drop outliers) → rolling → forecast 30d → ROP |

**Entrada:** `data/raw/online_retail_II.csv` (gitignored) o sample.  
**Salida principal:** `data/processed/inventory_reorder_recommendations.csv`.

```bash
python -m inventario_ecommerce.modeling.train
python -m inventario_ecommerce.modeling.predict
```
