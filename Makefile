.PHONY: requirements notebook notebook2 notebook3 train predict

requirements:
	pip install -r requirements.txt

notebook:
	jupyter notebook notebooks/01_carga_limpieza_retail.ipynb

notebook2:
	jupyter notebook notebooks/02_forecast_reorder_baseline.ipynb

notebook3:
	jupyter notebook notebooks/03_demand_hygiene.ipynb

train:
	python -m inventario_ecommerce.modeling.train

predict:
	python -m inventario_ecommerce.modeling.predict
