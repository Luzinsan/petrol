from core.data import Dataset
from core.train import Train
import os
import sys
from yaml import load, Loader



if __name__ == "__main__":  

    config_file = os.environ['CONF_LGBM']
    configs = load(open(config_file, "r"), Loader=Loader)['train']

    df = Dataset(name='1250', verbose=False, 
                 features_names=configs['features'], targets_names=configs['target'])
    df.load(f"{os.environ['PREPARED_DATA_PATH']}/{sys.argv[1]}/1250.csv")

    train = Train(df, config_file)
    train.optimize()
