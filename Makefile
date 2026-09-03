.PHONY: requirements notebook

requirements:
	pip install -r requirements.txt

notebook:
	jupyter notebook notebooks/01_carga_limpieza_retail.ipynb
