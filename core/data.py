import os
import itertools
import datetime
import builtins
import re

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from ydata_profiling import ProfileReport

from statsmodels.nonparametric.smoothers_lowess import lowess
from phik.report import plot_correlation_matrix
from scipy import signal

from sklearn import preprocessing

from core.configs import *

# from IPython import get_ipython
# ipython = get_ipython()

# slice функция по индексу для словарей
slice = lambda d, start=0, stop=None, step=1: dict(itertools.islice(d.items(), start, stop, step))


class Dataset:

    #region base
    def __init__(self, df: pd.DataFrame=None, verbose=VERBOSE, cudf=False):
        self.verbose = verbose
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
            self.original = df.copy()
        # if cudf:
        #     ipython.magic("load_ext cudf.pandas")
        

    def load(self, path, dropna=True, parse_dates=None):
        print("Чтение датасета...") if self.verbose else None
        match path.split('.')[-1]:
            case 'xlsx':
                self.df: pd.DataFrame = pd.read_excel(path, parse_dates=parse_dates)
            case 'csv':
                self.df: pd.DataFrame = pd.read_csv(path, parse_dates=parse_dates)
        
        self.original = self.df.copy()
        self.df.columns = [col.replace('|', ',') for col in self.df]
        if dropna:
            print(f"Удаленые записи\n{self.df.isna().sum()}") if self.verbose else None
            self.df = self.df.dropna().reset_index(drop=True)
        print("Завершено успешно") if self.verbose else None

    def __iter__(self):
        return self.cols.__iter__()
    
    def astype(self, columns, type):
        columns = self.columns(columns)
        for col in columns:
            self.df[col] = self.df[col].astype(type)
    
    @property
    def cols(self):
        return self.df.columns
    
    @property
    def data(self):
        return self.df.iloc[:,1:-1]

    
    @property
    def target(self):
        return self.df.iloc[:,-1]
    

    def columns(self, pattern: list|str):
        match type(pattern):
            case builtins.list | pd.Index:
                return [col for col in pattern if col in self.cols]
            case builtins.str:
                return [col for col in self.df if re.search(pattern, col)]
            case _:
                if pattern==None:
                    return self.cols
                else:
                    raise TypeError("Передан некорректный параметр 'pattern'")
                
    def drop(self, columns):
        columns = self.columns(columns)
        self.df.drop(columns, axis=1, inplace=True)
    
    def save(self, filename=None, extension='csv'):
        match extension:
            case 'csv':
                self.df.to_csv(filename, index=False)
            case 'xslx':
                self.df.to_excel(filename, index=False)

    def __insert_or_replace(self, column, insert, method, data):
        if insert:
            index = self.cols.get_loc(column) + 1
            new_col = column \
                    if column.endswith(method) \
                    else f'smoothed,{column},{method}'
            print('Вставка нового столбца: ', new_col) if self.verbose else None
            try:
                self.df.insert(index, new_col, data)
            except ValueError as err:
                print(err) if self.verbose else None
                self.df.loc[:, new_col] = data
        else:
            self.df.loc[:, column] = data

    #endregion

    #region time processing
    def parse_datetime(self, parse_date=True, drop_date=DROP_OLD_COLUMNS, 
                       parse_time=True, drop_time=DROP_OLD_COLUMNS, 
                       replace_time_0_to_24=False):
        """
                Расчленяет фичу "Дата" на 3 колонки: ['Год','Месяц','День']
                и преобразует фичу "Время" (datetime) в фичу "Час" (int)
            parse_date: bool (default True) - флаг на разрешение расчленения фичи "Дата"
            drop_date: bool (default True) - флаг на удаление фичи "Дата" после расчленения
            parse_time: bool (default True) - флаг на разрешение преобразования фичи "Время"
            drop_time: bool (default True) - флаг на удаление фичи "Время" после преобразования
            replace_time_0_to_24: bool (default False) - флаг замены 0 (полночь) на 24
        """
        print("Парсинг Даты и Времени...") if self.verbose else None
        if parse_date:
            loc=self.cols.get_loc('Дата')
            self.df.insert(loc, 'День', self.df['Дата'].dt.day)
            self.df.insert(loc, 'Месяц', self.df['Дата'].dt.month)
            self.df.insert(loc, 'Год', self.df['Дата'].dt.year)
            
        if parse_time:
            self.df.insert(self.cols.get_loc('Время'), 'Час', self.df['Время'].apply(lambda x: x.hour))
            if replace_time_0_to_24:
                self.df.loc[self.df['Час'] == 0, 'Час'] = 24
        if drop_date:
            self.df.drop(columns='Дата', inplace=True)
        if drop_time:
            self.df.drop(columns='Время', inplace=True)

    def set_lag(self, lag=0, drop=DROP_OLD_COLUMNS, target='Добыча воды за 2 ч ,м3'):
        """
                Добавления смещения значений по оси 0 (по строкам)
                Отрительные значения -> смещение вверх, положительные -> вниз
            lag: (default 0) int - значение смещение (лага, шага) по строкам
            drop: bool (default False) - флаг на удаление смещаемой фичи (оригинальной)
            target: str (default Добыча воды за 2 ч ,м3) - название фичи для сдвига
        """
        print(f"Смещение признака '{target}' на lag {lag}") if self.verbose else None
        if lag:
            self.df = pd.concat([self.df, 
                                self.df[target].shift(lag).
                                rename(f"{target}, лаг {lag}")], axis=1).dropna().reset_index(drop=True)
            if drop:
                self.df.drop(columns=target, inplace=True)
            
            print(f"'{target}' смещено на {lag}") if self.verbose else None
        else:
            print('Смещение установлено в 0. Пропускаем...') if self.verbose else None

    def filter_by_hours(self, hours: list=[8, 20]):
        """
                Фильтрует датасет по `hours` часам
                Необходим признак 'Час'
                *Метод parse_datetime достает фичу 'Час'*
            hours: list (default [8, 20]) - int значения часов для фильтрации
        """
        print(f"Фильтрация по {hours} часам") if self.verbose else None
        if 'Час' in self.cols:
            self.df = self.df[self.df['Час'].isin(hours)].reset_index(drop=True)
            print(self.df['Час'].value_counts()) if self.verbose else None
        else:
            print("Признак 'Час' не найден")

    def convert_datetime(self, drop=DROP_OLD_COLUMNS):
        datetimes = ['Год', 'Месяц','День','Час']
        self.df.insert(0, TIME_AXIS, self.df[datetimes].apply(
            lambda x: f'{x.iloc[0]:04}-{x.iloc[1]:02}-{x.iloc[2]:02} {x.iloc[3]:02}:00', 
            axis = 1)) 
        self.df.sort_values(TIME_AXIS, inplace=True)
        if drop:
            self.df.drop(columns=datetimes, inplace=True)

    #endregion

    #region preprocessing
    def new_feature(self, columns, newcol, agg_f:str='max', drop=DROP_OLD_COLUMNS):
        columns = self.columns(columns)
        match agg_f:
            case 'max':
                self.df[newcol] = self.df[columns].max(axis=1)
            case 'median':
                self.df[newcol] = self.df[columns].median(axis=1)
            case 'mean':
                self.df[newcol] = self.df[columns].mean(axis=1)
        if drop:
            self.drop(columns)


    def recovery_outliers(self, columns, save_interval, insert=True):
    
        columns = self.columns(columns)
        print("Восстановление выбросов...") if self.verbose else None
        for idx, col in enumerate(columns):
            mask = (
                    (self.df[col].between(*save_interval) )
                )
            
            data = pd.Series(np.where(mask, self.df[col], np.nan))
            print("Выбросы: ", self.df.loc[~mask, col]) if self.verbose else None
            data = data.ffill()
            self.__insert_or_replace(col, insert, 'outlier', data)
            


    def smooth(self, columns, frac=0.001, window=5, polyorder=3, insert=INSERT_NEARBY, method='lowess'):
        """
            Сглаживание значений в указанных признаках
            Поддерживаемые методы:
            - lowess - Сглаживание по методу Лоусса
            - rolling - Скользящее среднее окно
            - savgol_filter (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) - Фильтр Савицкого-Голея 
                - подгоняет последующие окна смежных данных с полиномом низкого порядка
        """
        columns = self.columns(columns)
        print("Выполняется сглаживание значений рядов: ", columns) if self.verbose else None
        for col in columns:
            match method:
                case 'lowess':
                    smoothed = lowess(self.df[col], range(len(self.df)), frac=frac)[:, 1]
                case 'rolling':
                    smoothed = self.df[col].rolling(window=window).mean()
                case 'savgol_filter':
                    smoothed = signal.savgol_filter(self.df[col],
                               window, # window size used for filtering
                               polyorder) # order of fitted polynomial
                case _:
                    raise ValueError("Указан неподдерживаемый метод")

            self.__insert_or_replace(col, insert, method, smoothed)
           

    def zscore(self, s, window, thresh=3, return_all=False, coeff=1.0):
        roll = s.rolling(window=window, min_periods=1)
        avg = roll.mean()
        max = roll.max()
        min = roll.min()
        std = roll.std(coeff)
        z = s.sub(avg).div(std)   
        m = z.between(-thresh, thresh)
        
        if return_all:
            return s.where(m, avg), max, min
        return s.where(m, avg)

    def scale(self, columns, method='standard', insert=False):
        columns = self.columns(columns)
        print("Выполняется масштабирование значений признаков: ", columns) if self.verbose else None
        match method:
            case 'standard':
                scaled = preprocessing.StandardScaler().fit_transform(self.df[columns])
            case 'minmax':
                scaled = preprocessing.MinMaxScaler().fit_transform(self.df[columns])
            case _:
                raise ValueError("Указан неподдерживаемый метод")
        self.__insert_or_replace(columns, insert, method, scaled)
            

        

    #endregion

    #region plots  
    def __plot_template(self, fig: go.Figure, title='', filepath=FILEPATH, show=SHOW_PLOTS, figsize=FIGSIZE, append=APPEND_TO_EXISTS):
        fig.update_layout(template=PLOT_THEME, height=figsize[0], width=figsize[1], title=title)
        if filepath:
            filepath = filepath if filepath.endswith('.html') else f'{filepath}.html'
            if not os.path.exists(filepath) or not append:
                fig.write_html(filepath)
            else:
                with open(filepath, 'a') as f:
                    f.write(fig.to_html(full_html=False, include_plotlyjs=False))
            print("Файл сгенерирован: ", filepath) if self.verbose else None
        if show:
            fig.show()

    def report(self, columns=None, filepath=FILEPATH, show=SHOW_PLOTS):
        columns = self.columns(columns)
        report = ProfileReport(self.df[columns], 
                      title="Profiling Report")
        if filepath:
            report.to_file(output_file=filepath)
        if show:
            return report    

    def time_series(self, columns=None, appendix_cols=None, log=LOG, time_axis=TIME_AXIS,
                    **kwargs): # filepath, show, figsize, append
        columns = self.columns(columns)
        if appendix_cols:
            columns += appendix_cols
        if time_axis not in self.cols:
            columns.append(time_axis)
        print("График отображает признаки: ", columns) if self.verbose else None
    
        try:
            fig = px.line(self.df, x=time_axis, y=columns,
                          log_x=log[0], log_y=log[1])
            self.__plot_template(fig, **kwargs)
        except BaseException as err:
            print(err)

        
    def scatter_matrix(self, columns=None, **kwargs):
        columns = self.columns(columns)
        fig = px.scatter_matrix(self.df[columns])
        fig.update_traces(diagonal_visible=False, showlowerhalf=False)
        self.__plot_template(fig, **kwargs)

    def difference_with_smoothed(self, column, time_axis=TIME_AXIS, method='rolling',mode='markers',
                                 **kwargs): # filepath, show, figsize, append
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df[time_axis],
            y=self.df[column],
            marker=dict(size=2, color='black',),
            opacity=0.25,
            name=column
        ))
        xaxis = self.df[time_axis]
        try:
            ylabel = self.columns(column+","+method)[0]
            yaxis = self.df[ylabel]
        except BaseException as err:
            print(err, f'ylabel: {self.columns(column)}')
        
        fig.add_trace(go.Scatter(
            x=xaxis,
            y=yaxis,
            marker=dict(
                size=6,
                color='royalblue',
                symbol='circle-open'
            ),
            name=ylabel
        ))

        if mode:
            fig.add_trace(go.Scatter(
                x=xaxis,
                y=yaxis,
                mode=mode,
                marker=dict(
                    size=6,
                    color='mediumpurple',
                    symbol='triangle-up'
                ),
                name='Smoothed scatter'
            ))

        self.__plot_template(fig, **kwargs)


    def corr_matrix(self, columns=None, target=TARGET, title='',
                    method='pearson', textfont_size=TEXTFONT_SIZE, 
                    **kwargs): # filepath, show, figsize, append
        columns = self.columns(columns)
        if target not in columns:
            columns.append(target)
       
        if target:
            corr = pd.concat([pd.DataFrame(self.df[columns].corr('pearson')[target]).rename(columns={target:'pearson'}).T,
                            pd.DataFrame(self.df[columns].corr('kendall')[target]).rename(columns={target:'kendall'}).T,
                            pd.DataFrame(self.df[columns].corr('spearman')[target]).rename(columns={target:'spearman'}).T,
                            pd.DataFrame(self.df[columns].phik_matrix()[target]).rename(columns={target:'phik'}).T]) 
        elif method=='phik': 
            corr = self.df[columns].phik_matrix()
        else:
            corr = self.df[columns].corr(method) 
       
        fig = px.imshow(corr, text_auto=True,  color_continuous_scale=COLORMAP, title=f'{title}: {target if target else method}')
        fig.update_traces(textfont_size=textfont_size,  texttemplate = "%{z:.2f}")
        self.__plot_template(fig, **kwargs)

    #endregion


