from configs import *
from data import Dataset
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

class Train:

    def __init__(self, df: Dataset, 
                 metrics=["r2", "mae", "mse", "rmse"], 
                 directions=['maximize', 'minimize', 'minimize', 'minimize']):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(df.data, df.target, 
                                                          test_size=0.2, random_state=2024)
        self.base_params = {
            "objective": "regression",
            "metrics": metrics,
            "verbosity": -1,
            "bagging_freq": 1,
            "random_state":2024,
            "device_type":'gpu',
            "force_col_wise":True,
        }
        self.metrics_list = metrics
        self.directions = directions

    def train_dataloader(self):
        return self.X_train, self.y_train

    def val_dataloader(self):
        return self.X_val, self.y_val
    
    def test_dataloader(self):
        return self.X_val

    def metrics(self, y_hat): # hardcode
        r2 = r2_score(self.y_val, y_hat)
        mae = mean_absolute_error(self.y_val, y_hat)
        mse = mean_squared_error(self.y_val, y_hat)
        rmse = root_mean_squared_error(self.y_val, y_hat)
        return r2, mae, mse, rmse

    def objective(self, trial):
        params = {
            **self.base_params,
            "n_estimators": trial.suggest_int("n_estimators", 100, 10**3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 100),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(*self.train_dataloader())
        predictions = model.predict(self.X_val)

        return self.metrics(predictions)
        
    def optimize(self, study_name, n_trials=100):
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name=study_name,
            directions=self.directions,
            load_if_exists=True
        )
        study.set_metric_names(self.metrics_list)
        study.optimize(self.objective, n_trials=n_trials)
        self.best_trials = study.best_trials
    
    def train(self, **params):
        self.lgbm = lgb.LGBMRegressor(**params)
        self.lgbm.fit(*self.train_dataloader())
        self.pred = self.lgbm.predict(self.X_val)
        return self.metrics(self.pred)
    
    def try_all_trials(self, n_estimators=10000):
        for trial in self.best_trials:
            trial.params.update(self.base_params)
            trial.params['n_estimators'] = n_estimators
            print(trial.params)
            print("R2 = {}; MAE = {}; MSE = {}; RMSE = {}".format(*self.train(**trial.params)))
            self.importances()

    def importances(self):
        lgb.plot_importance(self.lgbm, 
                            importance_type="gain", 
                            figsize=(10,10), title="LightGBM Feature Importance (Gain)")
        plt.show()

    def plot(self):
        plt.rc('font', size=11)
        plt.figure(figsize=(5, 4))
        plt.scatter(y_val, pred_lgbm, label="LGBM")

        plt.plot(y_val, y_val, label="True values", color="red")
        plt.legend();

        plt.xlabel("True values");
        plt.ylabel("Predicted values");
    
        