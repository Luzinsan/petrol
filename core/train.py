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
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle

import mlflow
from mlflow.models import infer_signature


class Train:

    def __init__(self, df: Dataset, model_name='lgbm',
                 metrics=["r2", "mae", "mse", "rmse"], 
                 directions=['maximize', 'minimize', 'minimize', 'minimize'],
                 predict=False, train=False,
                 test_size=0.2):
        self.dataset = df
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
        return self.train(False, **params)  
        
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
    
    def predict(self, track=False):
        if self.model_name == 'catboost':
            test_pool = catboost.Pool(self.X_val, label=self.y_val)
            self.pred = self.model.predict(test_pool)
        self.pred = self.model.predict(self.X_val)
        if track:
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
        

    def train(self, track=False, **params):
        
        self.model_init(self.model_name)
        self.model.set_params(**params)
        self.model.fit(*self.train_dataloader())

        # for catboost
        if self.model_name == 'catboost':
            test_pool = catboost.Pool(self.X_val, label=self.y_val)
            self.pred = self.model.predict(test_pool)
            return catboost.eval_metrics(test_pool, metrics=self.metrics, plot=True)
        self.predict(track)
        return self.metrics(self.pred, track) 
    
    def try_all_trials(self, n_estimators=10000):
        mlflow.end_run()
        mlflow.set_experiment(self.train_name)
        with mlflow.start_run(run_name=self.train_name):
            for trial in self.best_trials:
                self.train_with_trial(trial)

    def train_with_trial(self, trial, n_estimators=1000):
        trial.params.update(self.base_params)
        trial.params['n_estimators'] = n_estimators
        mlflow.set_experiment(self.train_name)
        with mlflow.start_run(nested=True):
            run_path = join(mlflow.active_run().info.artifact_uri, self.train_name)
            mlflow.log_params(trial.params)
            print(trial.params)
            print("R2 = {}; MAE = {}; MSE = {}; RMSE = {}".format(*self.train(True, **trial.params)))
            self.importances(run_path)
            self.plot_preds(run_path)
            self.dataset.corr_matrix(self.dataset.cols[1:], self.dataset.target.name, 
                                     title=f'Корреляционная матрица с таргетом {self.dataset.target.name}',
                                     filepath=join(run_path, 'corr_matrix.html'),
                                     mlflow_track=True)

    def importances(self, run_path):
        filepath = join(run_path, 'feature_importances.html')
        if self.model_name == 'lgbm':
            self.model.importance_type = "gain"
            importances = pd.DataFrame({"feature_names": self.X_train.columns, 
                                        "values": self.model.feature_importances_})\
                                            .sort_values('values', ascending=False).iloc[:10,:]
            fig = px.pie(importances, names="feature_names", values="values")
            fig.update_traces(textposition='inside', textinfo='percent+label')
        else:
            catboost.get_feature_importance(type='PredictionValuesChange')
        
        Dataset.plot_template(fig, filepath, 
                              title="LightGBM Feature Importance (Gain)", 
                              mlflow_track=True, height=500, width=1000)

        

    def plot_preds(self, run_path):
        QQ = pd.DataFrame({"Validate": self.y_val, "Predicted": self.pred})

        filepath = join(run_path, 'True_predicted_time_series.html')
        fig = px.line(QQ, x=np.arange(len(self.y_val)), y=['Validate', 'Predicted'], log_y=True)
        Dataset.plot_template(fig, filepath, 
                              title=f"True and predicted values of {self.y_val.name}", 
                              mlflow_track=True, height=500, width=1000)

        filepath = join(run_path, 'QQ_plot.html')
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=QQ['Validate'],
                y=QQ['Predicted'],
                mode='markers',
                marker=dict(size=10, color='blue',),
                name='Predicted'
            ))
        fig.add_trace(go.Scatter(
                x=QQ['Validate'],
                y=QQ['Validate'],
                name='Validate'
            ))
        Dataset.plot_template(fig, filepath, 
                              title=f"QQ Plot of {self.y_val.name}", 
                              mlflow_track=True, height=400, width=1000)
    
