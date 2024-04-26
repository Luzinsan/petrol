export YEAR=2022
export RAW_DATA_PATH=/home/prog3/notebooks/petrol/data/raw
export INITIAL_DATA_PATH=/home/prog3/notebooks/petrol/data/initial
export PREPARED_DATA_PATH=/home/prog3/notebooks/petrol/data/prepared
export CONF_LGBM=/home/prog3/notebooks/petrol/configs/lightgbm_conf.yml

prepare_stage1:
	poetry run python data/get_data.py

prepare_stage2:
	poetry run python data/prepare_data.py

prepare_data: prepare_stage1 prepare_stage2


optimization:
	# cd ~/notebooks/petrol/ ; poetry run optuna-dashboard sqlite:///db.sqlite3 --port 8089 & poetry run python ./piplines/optimize/lightgbm_opt.py
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/optimize/lightgbm_opt.py

training:
	# cd ~/notebooks/petrol/ ; poetry run mlflow ui --port 8090 & poetry run python ./piplines/train/lightgbm_train.py
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/train/lightgbm_train.py


custom_train:
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/train/lightgbm_custom_train.py