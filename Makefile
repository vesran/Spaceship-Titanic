
lab:
	@jupyter lab --ip=0.0.0.0 --allow-root

split:
	@python src/dataprep/split.py

submission:
	@python src/run.py