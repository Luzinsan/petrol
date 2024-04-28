from core.data import Dataset
from core.train import Train
import os
import sys
from yaml import load, Loader
import mlflow
from mlflow.pyfunc import PyFuncModel



if __name__ == "__main__":  

    config_file = os.environ['CONF_LGBM']
    configs = load(open(config_file, "r"), Loader=Loader)
    
    
    df = Dataset(name='1250', verbose=False, 
                 features_names=configs['train']['features'], targets_names=configs['train']['target'])
    df.load(f"{os.environ['PREPARED_DATA_PATH']}/{sys.argv[1]}/1250.csv")

    train = Train(df, config_file, mode='test')

    run = mlflow.search_runs(experiment_names=[configs['train']['experiment_name']], 
                              filter_string=f"run_name='{configs['train']['train_name']}'").iloc[0].run_id
    train.test(run)

   

    