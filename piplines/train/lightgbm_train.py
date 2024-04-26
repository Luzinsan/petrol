from core.data import Dataset
from core.train import Train
import os
import optuna




if __name__ == "__main__":  
    df = Dataset(name='1250', verbose=False)
    df.load(f"{os.environ['PREPARED_DATA_PATH']}/{os.environ['YEAR']}/1250.csv")

    train = Train(df, os.environ['CONF_LGBM'])
    train.train_best_trial()

    