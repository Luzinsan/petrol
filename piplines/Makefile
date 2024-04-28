export RAW_DATA_PATH=/home/prog3/notebooks/petrol/data/raw
export INITIAL_DATA_PATH=/home/prog3/notebooks/petrol/data/initial
export PREPARED_DATA_PATH=/home/prog3/notebooks/petrol/data/prepared
export CONF_LGBM=/home/prog3/notebooks/petrol/configs/lightgbm_conf.yml


prepare_data_2022:
	poetry run python data/get_data.py 2022
	poetry run python data/prepare_data.py 2022

prepare_data_2023:
	poetry run python data/prepare_data.py 2023

# optuna-dashboard sqlite:///db.sqlite3
# mlflow ui

optimization_2022:
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/optimize/lightgbm_opt.py 2022

optimization_2023:
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/optimize/lightgbm_opt.py 2023

training_2022:
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/train/lightgbm_train.py 2022

training_2023:
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/train/lightgbm_train.py 2023


predicting_2023: 
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/predict/lightgbm_predict.py 2023

predicting_2022: 
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/predict/lightgbm_predict.py 2022


full_pipline_from_2022: prepare_data_2022 optimization_2022 training_2022 prepare_data_2023 predicting_2023
full_pipline_from_2023: prepare_data_2023 optimization_2023 training_2023 prepare_data_2022 predicting_2022

custom_train:
	cd ~/notebooks/petrol/ ; poetry run python ./piplines/train/lightgbm_custom_train.py