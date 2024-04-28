from core.configs import *
from core.data import Dataset
import lightgbm as lgb
import catboost

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from functools import reduce
from pathlib import Path


import mlflow
from mlflow.models import infer_signature
from yaml import load, Loader


class Train:

    def __init__(self, df: Dataset,
                 config_file:str|Path,
                 mode='train'):
        
        with open(config_file, "r") as conf:
            self.configs = load(conf, Loader=Loader)

        np.random.seed(2024)
        self.df: Dataset = df
        
        if mode=='train':
            self.X_train_index, self.X_val_index, self.y_train_index, self.y_val_index = \
                                train_test_split(self.df.data.index, self.df.target.index,
                                                test_size=self.configs['base']["validation_size"])
            print(f'train shape: {self.X_train_index.shape}, validate shape: {self.X_val_index.shape}')
        elif mode=='test':
            self.X_val_index, self.y_val_index = self.df.data.index, self.df.target.index
            print(f'test shape: {self.X_val_index.shape}')
        else:
            self.X_train_index, self.X_val_index, self.y_train_index, self.y_val_index = \
                                train_test_split(self.df.data.index, self.df.target.index,
                                                test_size=self.configs['base']["validation_size"])
            self.X_train_index, self.X_test_index, self.y_train_index, self.y_test_index = \
                                train_test_split(self.X_train_index, self.y_train_index,
                                            test_size=self.configs['base']["test_size"])
            print(f'train shape: {self.X_train_index.shape}, test shape: {self.X_test_index.shape}, validate shape: {self.X_val_index.shape}')
    
        self.model_init()
        mlflow.set_tracking_uri('mlruns')

    def model_init(self):
        map = {'lightgbm': lgb.LGBMRegressor, 'catboost': catboost.CatBoostRegressor}
        try:
            self.model = map[self.configs['base']['model']]()
        except KeyError as err:
            raise KeyError("Неподдерживаемое название модели")
        


    def metrics(self, y_hat, track=False):
        metric_values = dict()
        map = {'r2': r2_score , 'mae': mean_absolute_error, 
               'mse':mean_squared_error,'rmse':root_mean_squared_error}
        
        for metric in self.configs['base']['metrics']:
            metric_values[metric] = map[metric.lower()](self.df.target[self.y_val_index], 
                                                        y_hat)
        # Log the loss metric
        if track:
            mlflow.log_metrics(metric_values)
        return list(metric_values.values())

    def objective(self, trial):
        params = dict()
        # Простите за мой французский, но здесь пиздец
        map = {'int':trial.suggest_int,'float':trial.suggest_float}
        optimizing_params = reduce(
            lambda a, b: {**a, **b},
            [
                {
                    param: map[_type](param, *args['range'], **args.get('args')) 
                            if args.get('args') else map[_type](param, *args['range'])
                    for param, args in _params.items()
                } for _type, _params in self.configs['optimize'].items()
            ])
        params = {
            **self.configs['base'],
            **optimizing_params,
        }
        return self.train(False, **params)  
        
    def optimize(self):
        n_trials=self.configs['optimize'].pop('n_trials')
        study = optuna.create_study(
            storage=self.configs['optimize'].pop('storage'),
            study_name=self.configs['optimize'].pop('study_name'),
            directions=self.configs['optimize'].pop('directions'),
            load_if_exists=self.configs['optimize'].pop('load_if_exists')
        )
        
        study.set_metric_names(self.configs['base']['metrics'])
        study.optimize(self.objective, n_trials=n_trials)
        self.best_trials = study.best_trials
    
    def predict(self, track=False):
        pred = self.model.predict(self.df[self.X_val_index])
        return pred
            

    def train(self, track=False, **params):
        self.model_init()
        self.model.set_params(**params)
        self.model.fit(self.df[self.X_train_index], self.df.target[self.y_train_index])     
         
        self.pred = self.predict(track)
        if track:
            map = {'lightgbm': (mlflow.lightgbm.log_model, 'lgb_model'), 
                   'catboost': (mlflow.catboost.log_model, 'cb_model')}
            logger_dict = dict(
                            artifact_path=self.configs['train']['train_name'],
                            signature=infer_signature(self.df[self.X_train_index], self.pred), # Infer the model signature
                            input_example=self.df[self.X_train_index],
                            registered_model_name=self.configs['base']['model'],
                            )
            # Log the model
            model = map[self.configs['base']['model']]
            model[0](**{model[1]: self.model, **logger_dict})


        return self.metrics(self.pred, track) 
    
    def test(self, run_id):
       
        artifacts_path = mlflow.get_run(run_id).info.artifact_uri + '/test/'
        Path(artifacts_path).mkdir(parents=True, exist_ok=True)

        self.model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{self.configs['train']['train_name']}")

        self.pred = self.predict()
        metrics = { name:value for name, value in zip(self.configs['base']['metrics'], 
                                                  self.metrics(self.pred))}
        mlflow.log_dict(metrics, './test/test_metrics.json', run_id=run_id)
        
        self.df.corr_matrix(self.df.data.columns, self.df.target.name, 
                            title=f'Корреляционная матрица на тестовой выборке с таргетом {self.df.target.name}',
                            filepath=join(artifacts_path, 'corr_matrix_for_test.html'),
                            mlflow_track=True, run_id=run_id)
        self.plot_preds(artifacts_path, run_id=run_id)

        
        
    

    def train_with_params(self, params):
        mlflow.set_experiment(self.configs['train']['experiment_name'])
        with mlflow.start_run(nested=True, run_name=self.configs['train']['train_name']):
            
            mlflow.log_params(params)
            mlflow.log_input(
                mlflow.data.from_pandas(
                    self.df.df[1:], name=self.df.name, targets=self.configs['train']['target']
                ), context="training") 
            print(params)
            print({ name:value for name, value in zip(self.configs['base']['metrics'], 
                                                      self.train(True, **params))})
            run_path = mlflow.active_run().info.artifact_uri
            self.log_plots(run_path)
            

    def log_plots(self, run_path):
        self.importances(run_path)
        self.plot_preds(run_path)
        self.df.corr_matrix(self.df.data.columns, self.df.target.name, 
                            title=f'Корреляционная матрица с таргетом {self.df.target.name}',
                            filepath=join(run_path, 'corr_matrix.html'),
                            mlflow_track=True)
        

    def train_best_trial(self):
        study: optuna.Study = optuna.load_study(study_name=self.configs['optimize']['study_name'],
                                                storage=self.configs['optimize']['storage'])
        scores = np.array([trial.values for trial in study.best_trials])
        
        col_idx = self.configs['base']['metrics'].index(
            self.configs['train']['target_score'])
        direction = self.configs['optimize']['directions'][col_idx]
        np_func = np.argmax if direction=='maximize' else np.argmin
        
        params= study.best_trials[np_func(scores[:,col_idx])].params
        params.update(self.configs['base'])
        params['n_estimators'] = self.configs['train'].pop('n_estimators')
        self.train_with_params(params)
        
        

    def importances(self, run_path):
        
        self.model.importance_type = "gain"
        if self.configs['base']['model'] == 'lightgbm':
            importances = self.model.feature_importances_
        elif self.configs['base']['model'] == 'catboost':
            importances = self.model.get_feature_importance()
        
        
        importances = pd.DataFrame({"feature_names": self.df.data.columns, 
                                    "values": importances})#\
                                        #.iloc[:,:]
        fig = px.pie(importances, names="feature_names", values="values")
        fig.update_traces(textposition='inside', textinfo='percent+label')

        filepath = join(run_path, 'feature_importances.html')
        Dataset.plot_template(fig, filepath, 
                              title=f"{self.configs['base']['model']} Feature Importance (Gain)", 
                              mlflow_track=True, height=800, width=1000)

        

    def plot_preds(self, run_path, run_id=None):
        QQ = pd.DataFrame({"Validate": self.df.target[self.y_val_index], "Predicted": self.pred})

        filepath = join(run_path, 'True_predicted_time_series.html')
        fig = px.line(QQ, x=np.arange(len(self.y_val_index)), y=['Validate', 'Predicted'], log_y=True)
        Dataset.plot_template(fig, filepath, 
                              title=f"True and predicted values of {self.y_val_index.name}", 
                              mlflow_track=True,run_id=run_id, height=500, width=1500)

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
                              title=f"QQ Plot of {self.y_val_index.name}", 
                              mlflow_track=True, run_id=run_id, height=500, width=1000)
    
