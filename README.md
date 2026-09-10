# Optimización de Inventario E-commerce

Convierte el histórico de facturas de un retailer online en una **tabla de reposición por SKU**: qué demanda se espera, a qué nivel de stock hay que volver a pedir y cuánto pedir. Y después simula esa política sobre datos que no vio, para saber cuántos quiebres evita y cuánto stock cuesta.

Datos: [Online Retail II (UCI / Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci), ~1.07M líneas de factura entre 2009 y 2011.

## El entregable

`data/processed/inventory_reorder_recommendations.csv` — 2,963 SKUs activos. Las cinco primeras filas por demanda prevista:

| SKU | Descripción | ABC | Forecast (ud/día) | Modelo | Punto de reorden | Pedido sugerido |
|-----|-------------|:---:|------:|:---:|------:|------:|
| `22197` | Popcorn Holder | A | 259.9 | ets | 4,960 | 1,820 |
| `23084` | Rabbit Night Light | A | 218.6 | ets | 4,382 | 1,531 |
| `22086` | Paper Chain Kit 50's Christmas | A | 189.9 | ets | 3,923 | 1,330 |
| `84077` | World War 2 Gliders Asstd Designs | A | 146.9 | ma30 | 3,258 | 1,028 |
| `22578` | Wooden Star Christmas Scandinavian | A | 145.7 | ma30 | 3,173 | 1,020 |

- **Punto de reorden** — nivel de stock al que lanzar el pedido, stock de seguridad incluido.
- **Pedido sugerido** — demanda del ciclo de revisión (7 días), o `target − on_hand − on_order` si se aporta el stock físico.
- **Modelo** — `ets` en los 97 SKUs del top clase A donde Holt-Winters ajusta; `ma30` en el resto.

## Pipeline

1. **Limpieza** — nulos, devoluciones y líneas que no son producto físico (portes, ajustes, tests)
2. **ABC** — 80% / 95% de las ventas acumuladas del último trimestre
3. **Demanda diaria** — calendario continuo por SKU con los días sin venta a 0, cap al percentil 99 **de cada clase ABC** y exclusión de one-shots
4. **Forecast** — media móvil 30 días en todo el catálogo; Holt-Winters en el top 100 clase A
5. **Reorden** — `ROP = forecast_daily × LT + z × σ_30d × √LT`, con lead time 14 d, revisión 7 d y **z = 3.0** (coste mínimo bajo margen 40% / posesión 25%/año)
6. **Simulación** — se repone día a día sobre el holdout para medir quiebres y stock inmovilizado
7. **Coste** — se valoran esas unidades con margen y tasa de posesión supuestos; el z de producción sale de ahí

Reglas de limpieza, parámetros y columnas de salida en detalle: [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Resultados

Backtest temporal: se entrena con todo el histórico hasta el 2011-11-09 y se evalúa en los 30 días siguientes. El MAE se mide sobre el calendario completo, así que los días sin venta también puntúan.

| Grupo | Demanda media | MAE media móvil | MAE ETS |
|-------|------:|------:|------:|
| Catálogo (5,733 series) | 4.11 ud/día | 4.13 | — |
| Top 100 clase A | 53.03 ud/día | 40.47 | **37.04** |

A nivel de catálogo el error es del tamaño de la propia demanda: la mayoría de SKUs vende de forma esporádica y ahí una media móvil ya está cerca del techo de lo razonable. Donde hay volumen e historia —el top clase A— ETS reduce el MAE un 8.5% y gana en 69 de los 95 SKUs que ajustan.

No se reporta MAPE como métrica de decisión: con tantos días a cero en el denominador se dispara (~144%) y deja de ser informativo.

![Demanda real frente a media móvil y ETS](reports/figures/class_a_sku_forecast.png)

La media móvil es una recta: predice el mismo valor los 30 días y se queda corta durante la subida de Navidad. ETS sigue el nivel y el ciclo semanal, aunque nunca baja a cero en los días sin operación.

### Qué cuesta el nivel de servicio

El backtest mide el forecast; la simulación mide la decisión. Se repone día a día sobre el mismo holdout: llegan los pedidos, se sirve lo que hay, y lo que no se sirve se pierde. Cada política arranca con el stock que su propia regla exige.

| Política | Fill rate | Unidades perdidas | Stock medio (ud) | Días de cobertura |
|----------|------:|------:|------:|------:|
| Pedir la media (sin stock de seguridad) | 73.2% | 184,336 | 199,265 | 8.7 |
| z por clase antiguo (1.88 / 1.65 / 1.28) | 87.0% | 89,848 | 380,216 | 16.6 |
| z = 2.33 uniforme (99%) | 89.5% | 72,189 | 451,964 | 19.7 |
| **z = 3.0 (política actual)** | **91.6%** | **57,895** | **531,778** | **23.2** |

![Stock medio frente al fill rate de cada política](reports/figures/policy_service_tradeoff.png)

Pedir la media no es una política: deja sin servir una de cada cuatro unidades. La política antigua por clase compraba servicio a ~1.9 unidades de stock parado por unidad extra servida; seguir subiendo encarece el tramo. El análisis de coste (abajo) fijó el z de producción en **3.0** para las tres clases.

Incluso con z = 3.0 el fill rate se queda en 91.6%: el holdout es la subida de Navidad, la demanda real se va por encima de la media entrenada y con 14 días de lead time no da tiempo a reaccionar. Lo que falta a partir de aquí es nivel en el forecast, no más stock de seguridad.

### Qué cuesta en dinero

Las unidades no pesan igual: una no servida de un SKU caro y una parada de uno barato son costes distintos. Con el precio mediano del trimestre, un margen bruto del 40% y una tasa de posesión del 25%/año (supuestos: el dataset no trae coste de compra), se barre un grid de z y se elige el de menor coste total.

| z | Fill rate | Margen perdido | Capital inmovilizado | Coste total |
|--:|------:|------:|------:|------:|
| 0.00 (pedir la media) | 73.2% | 149,091 | 4,990 | 154,081 |
| 1.88 (política anterior, clase A) | 87.7% | 72,686 | 10,307 | 82,993 |
| **3.00 (adoptado)** | **91.6%** | **50,961** | **13,766** | **64,726** |

![Coste de quiebre, posesión y total frente a z](reports/figures/policy_cost_vs_service.png)

Sobre 30 días, inmovilizar una unidad cuesta ~2% de su coste; no servirla cuesta el 40% del precio. Por eso el óptimo se va al techo del grid: subir de z = 1.88 a z = 3.0 ahorra ~18k en el holdout. Las tres clases ABC eligen el mismo z = 3.0 —diferenciar el factor de seguridad por clase no se justifica en coste bajo estos supuestos—, y **esa es la política que usa `predict` hoy**.

Solo con márgenes muy finos (≤10%) y alta obsolescencia (≥50%/año) el óptimo baja. En el extremo (5% de margen, 100% de posesión) conviene z = 0. En retail razonable, el cuello de botella sigue siendo el forecast, no el stock de seguridad.

## Supuestos y limitaciones

Lo que este proyecto **no** resuelve, y conviene saber antes de leer los números:

- **No hay stock on-hand ni pedidos en curso.** El dataset son ventas, no inventario, así que el pedido sugerido es la demanda del ciclo de revisión. El pipeline ya calcula `target − on_hand − on_order` en cuanto se le da la fuente: basta un `data/raw/stock_on_hand.csv` con `StockCode`, `on_hand` y `on_order`.
- **Lead time y ciclo de revisión son supuestos** (14 y 7 días): no hay datos de proveedor.
- **El stock de seguridad asume normalidad.** `z × σ × √LT` supone demanda normal y lead time constante; la demanda real es intermitente, así que el nivel de servicio es aproximado, no garantizado.
- **El cap sigue recortando picos reales.** Por clase se queda en 315 ud/día (A), 183 (B) y 120 (C), frente a las 225 del cap global. Eso baja de 3,279 a 1,777 los días-SKU recortados en clase A, pero el máximo real de `22197` fueron 4,314 ud en un día: el MAE absoluto no es el error contra la demanda cruda. Los modelos se comparan sobre la misma serie recortada, así que la comparación entre ellos sí es válida.
- **Los sábados la tienda no factura** (400 facturas frente a 150-200k de cualquier otro día). Parte de los ceros del calendario no son demanda perdida, son días sin operación.
- **La simulación no tiene precios ni costes.** Compara unidades perdidas contra unidades en almacén. El módulo de coste (`modeling/economics.py`) valora esas unidades con un margen y una tasa de posesión **supuestos** (40% y 25%/año): el dataset trae precio de venta, no coste de compra. Tampoco modela backorders, capacidad de almacén ni lead time variable, y arranca cada política con su stock objetivo en la estantería: es un warm start, no una historia completa.

## Qué hay en este repo

| Pieza | Descripción |
|-------|-------------|
| `notebooks/01_carga_limpieza_retail.ipynb` | Carga, calidad de datos, limpieza y ventas del trimestre |
| `notebooks/02_forecast_reorder_baseline.ipynb` | ABC, backtest baseline y tabla de reorden |
| `notebooks/03_demand_hygiene.ipynb` | Qué cambia al rellenar ceros y recortar picos, y por qué el cap va por clase |
| `notebooks/04_forecast_class_a.ipynb` | Holt-Winters vs media móvil en el top clase A |
| `notebooks/05_policy_simulation.ipynb` | Quiebres vs stock inmovilizado de cada política |
| `notebooks/06_policy_cost.ipynb` | Coste en dinero: margen perdido vs capital inmovilizado |
| `inventario_ecommerce/` | Paquete: `config`, `dataset`, `features`, `modeling`, `plots` |
| `tests/` | Tests de limpieza, calendario, winsor, ABC, política, simulación y coste |
| `docs/data_dictionary.md` | Esquema de entrada, reglas de limpieza y columnas de la salida |
| `data/raw/sample_online_retail.csv` | Sample para probar el código sin bajar Kaggle |

Las notebooks están versionadas **con sus outputs**, así que se leen en GitHub sin ejecutarlas.

## Cómo ejecutarlo

```bash
pip install -e ".[dev]"
python -m pytest
```

Pipeline completo:

```bash
python -m inventario_ecommerce.modeling.train      # backtests y métricas
python -m inventario_ecommerce.modeling.predict    # tabla de reposición
python -m inventario_ecommerce.modeling.simulate   # quiebres vs stock por política
python -m inventario_ecommerce.modeling.economics  # coste en dinero y z óptimo
```

Las notebooks siguen el orden **01 → 02**; la **03** justifica la higiene de demanda, la **04** compara modelos en clase A, la **05** simula la política y la **06** la valora en dinero.

### Dataset completo

El CSV de ~90 MB **no** está en el repo:

1. Descarga [Online Retail II UCI (Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
2. Guárdalo como `data/raw/online_retail_II.csv`

Sin él, el sample local basta para validar el código.

## Estructura

```
inventario_ecommerce/     # paquete instalable
notebooks/                # 01 → 06, con outputs
tests/                    # pytest
docs/                     # diccionario de datos
data/                     # raw y processed (gitignored)
reports/figures/          # gráficos versionados
```

`train`, `predict`, `simulate` y `economics` dejan en `data/processed/` la tabla de reposición y los intermedios (ABC, features rolling, métricas de backtest, simulación y coste de la política).

## Stack

Python 3.13 · pandas · numpy · statsmodels · matplotlib · Jupyter · pytest

Licencia MIT (`LICENSE`).
