.PHONY: requirements test notebook notebook2 notebook3 notebook4 notebook5 notebook6 train predict simulate economics

requirements:
	pip install -e ".[dev]"

test:
	python -m pytest -q

notebook:
	jupyter notebook notebooks/01_carga_limpieza_retail.ipynb

notebook2:
	jupyter notebook notebooks/02_forecast_reorder_baseline.ipynb

notebook3:
	jupyter notebook notebooks/03_demand_hygiene.ipynb

notebook4:
	jupyter notebook notebooks/04_forecast_class_a.ipynb

notebook5:
	jupyter notebook notebooks/05_policy_simulation.ipynb

train:
	python -m inventario_ecommerce.modeling.train

predict:
	python -m inventario_ecommerce.modeling.predict

simulate:
	python -m inventario_ecommerce.modeling.simulate

economics:
	python -m inventario_ecommerce.modeling.economics

notebook6:
	jupyter notebook notebooks/06_policy_cost.ipynb
