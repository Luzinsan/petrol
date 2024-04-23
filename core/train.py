import matplotlib.axes
from core.configs import *
from core.data import Dataset
import lightgbm as lgb
import catboost

import optuna
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pickle

import mlflow
from mlflow.models import infer_signature


class Train:

    def __init__(self, df: Dataset, model_name='lgbm',
                 metrics=["r2", "mae", "mse", "rmse"], 
                 directions=['maximize', 'minimize', 'minimize', 'minimize'],
                 predict=False, train=False,
                 test_size=0.2):
        if predict:
            self.X_val, self.y_val = df.data, df.target
        elif train:
            self.X_train, self.y_train = df.data, df.target
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(df.data, df.target, 
                                                          test_size=test_size, random_state=2024)
        
        self.metrics_list = metrics
        self.directions = directions
        self.model_init(model_name)

    def model_init(self, model_name):
        self.model_name = model_name
        match model_name:
            case 'lgbm':
                self.base_params = {
                    "objective": "regression",
                    "metrics": self.metrics_list,
                    "verbosity": -1,
                    "bagging_freq": 1,
                    "random_state":2024,
                    "device_type":'gpu',
                    "force_col_wise":True,
                }
                self.model = lgb.LGBMRegressor()
            case 'catboost':
                self.base_params = {
                    'custom_metric':self.metrics_list,
                    'random_state':2024,
                    'task_type':'GPU',
                    'devices':'0-3',
                }
                self.model = catboost.CatBoostRegressor()


    def train_dataloader(self):
        return self.X_train, self.y_train

    def val_dataloader(self):
        return self.X_val, self.y_val
    
    def test_dataloader(self):
        return self.X_val

    def metrics(self, y_hat, track=False): # hardcode
        r2 = r2_score(self.y_val, y_hat)
        mae = mean_absolute_error(self.y_val, y_hat)
        mse = mean_squared_error(self.y_val, y_hat)
        rmse = root_mean_squared_error(self.y_val, y_hat)
        # Log the loss metric
        if track:
            mlflow.log_metrics(dict(r2=r2, mae=mae, mse=mse, rmse=rmse))
        return r2, mae, mse, rmse

    def objective(self, trial):
        params = dict()
        match self.model_name:
            case 'lgbm':
                params = {
                    **self.base_params,
                    "n_estimators": trial.suggest_int("n_estimators", 100, 10**2),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
                    "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 100),
                }
            case 'catboost':
                params= {
                    **self.base_params,
                    'iterations': trial.suggest_int("n_estimators", 50, 200),
                    'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    'depth': trial.suggest_int("min_data_in_leaf", 2, 30),
                    'l2_leaf_reg': trial.suggest_float("learning_rate", 0.1, 5),
                } 
        self.model_init(self.model_name)
        self.model.set_params(**params)
        self.model.fit(*self.train_dataloader())
            
        return self.metrics(self.model.predict(self.X_val))
        
    def optimize(self, study_name, n_trials=100):
        
        self.train_name = study_name
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name=study_name,
            directions=self.directions,
            load_if_exists=True
        )
        study.set_metric_names(self.metrics_list)
        study.optimize(self.objective, n_trials=n_trials)
        self.best_trials = study.best_trials
    
    def predict(self):
        if self.model_name == 'catboost':
            test_pool = catboost.Pool(self.X_val, label=self.y_val)
            self.pred = self.model.predict(test_pool)
        self.pred = self.model.predict(self.X_val)
        # Infer the model signature
        signature = infer_signature(self.X_train, self.pred)

        # Log the model
        model_info = mlflow.lightgbm.log_model(
            
            lgb_model=self.model,
            artifact_path=self.train_name,
            signature=signature,
            input_example=self.X_train,
            registered_model_name=self.train_name,
        )
        

    def train(self, **params):
        
        self.model_init(self.model_name)
        self.model.set_params(**params)
        self.model.fit(*self.train_dataloader())

        # for catboost
        if self.model_name == 'catboost':
            test_pool = catboost.Pool(self.X_val, label=self.y_val)
            self.pred = self.model.predict(test_pool)
            return catboost.eval_metrics(test_pool, metrics=self.metrics, plot=True)
        self.predict()
        return self.metrics(self.pred, True) 
    
    def try_all_trials(self, n_estimators=10000):
        mlflow.end_run()
        mlflow.set_experiment(self.train_name)
        with mlflow.start_run(run_name=self.train_name):
            for trial in self.best_trials:
                trial.params.update(self.base_params)
                trial.params['n_estimators'] = n_estimators
                mlflow.set_experiment(self.train_name)
                with mlflow.start_run(nested=True):
                    mlflow.log_params(trial.params)
                    print(trial.params)
                    print("R2 = {}; MAE = {}; MSE = {}; RMSE = {}".format(*self.train(**trial.params)))
                    self.importances()
                    self.plot_preds()
            

    def importances(self):
        # run_id = mlflow.active_run().info.run_id
        # filepath = join('mlruns', 
        #                 mlflow.get_parent_run(run_id).info.run_uuid(), 
        #                 run_id,
        #                 'feature_importances.png')
        if self.model_name == 'lgbm':
            lgb.plot_importance(self.model, 
                                importance_type="gain", 
                                figsize=(10,10), title="LightGBM Feature Importance (Gain)")
        else:
            catboost.get_feature_importance(type='PredictionValuesChange')
        
        # plt.savefig(filepath)
        # mlflow.log_artifact(filepath)
        return plt.show()

    def plot_preds(self):
        # run_id = mlflow.active_run().info.run_id
        # filepath = join('mlruns', 
        #                 mlflow.get_parent_run(run_id).info.run_uuid(), 
        #                 run_id,
        #                 'True_predicted_time_series.png')
        fig = px.line(x=np.arange(len(self.y_val)), y=[self.y_val, self.pred], 
                      labels={'wide_variable_0': 'Validate', 'wide_variable_1':'Predicted'}, 
                      log_y=True, width=1900, height=500).show()
        # fig.write_image(filepath)
        # mlflow.log_artifact(filepath)

        # filepath = join('mlruns', 
        #                 mlflow.get_parent_run(run_id).info.run_uuid(), 
        #                 run_id,
        #                 'True_Predicted.png')
        
        plt.rc('font', size=11)
        plt.figure(figsize=(5, 4))
        plt.scatter(self.y_val, self.pred, label="LGBM")

        plt.plot(self.y_val, self.y_val, label="True values", color="red")
        plt.legend()

        plt.xlabel("True values")
        plt.ylabel("Predicted values")

        # plt.savefig(filepath)
        # mlflow.log_artifact(filepath)
        return plt.show()
