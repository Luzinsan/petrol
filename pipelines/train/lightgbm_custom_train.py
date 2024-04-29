from core.data import Dataset
from core.train import Train
import os
import optuna




if __name__ == "__main__":  
    df = Dataset(name='1250', verbose=False)
    df.load(f"{os.environ['PREPARED_DATA_PATH']}/{os.environ['YEAR']}/1250.csv")

    train = Train(df, os.environ['CONF_LGBM'])
    params = dict(
                    model='lightgbm',
                    objective='regression',
                    metrics=['r2', 'mae', 'mse', 'rmse'],
                    verbosity=-1,
                    bagging_freq=1,
                    device_type='gpu',
                    force_col_wise=True,
                    
                    n_estimators=100,
                    learning_rate=0.026956697523389854,
                    num_leaves=410,
                    subsample=0.8850913452197978,
                    colsample_bytree=0.7702514421013131,
                    min_data_in_leaf=3
                    )
    train.train_with_params(params)

    