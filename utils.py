import pandas as pd
import os
import numpy as np
import itertools
import datetime
import plotly.graph_objects as go
import plotly.express as px
from ydata_profiling import ProfileReport
import re
import builtins
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import signal

sns_colormap = [[0.0, '#3f7f93'],
                [0.1, '#6397a7'],
                [0.2, '#88b1bd'],
                [0.3, '#acc9d2'],
                [0.4, '#d1e2e7'],
                [0.5, '#f2f2f2'],
                [0.6, '#f6cdd0'],
                [0.7, '#efa8ad'],
                [0.8, '#e8848b'],
                [0.9, '#e15e68'],
                [1.0, '#da3b46']]



class Dataset:

    def __init__(self, path, dropna=True, verbose=False):
        self.verbose = verbose
        print("Чтение датасета...") if self.verbose else None
        self.df: pd.DataFrame = pd.read_excel(path, parse_dates=['Дата'])
        self.original = self.df.copy()
        if dropna:
            print(f"Удаленые записи\n{self.df.isna().sum()}") if self.verbose else None
            self.df = self.df.dropna().reset_index(drop=True)
        print("Завершено успешно") if self.verbose else None

    def parse_datetime(self, parse_date=True, drop_date=True, 
                       parse_time=True, drop_time=True, 
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


    def set_lag(self, lag=0, delete_orig=False, colname_for_lag='Добыча воды за 2 ч |м3'):
        """
                Добавления смещения значений по оси 0 (по строкам)
                Отрительные значения -> смещение вверх, положительные -> вниз
            lag: (default 0) int - значение смещение (лага, шага) по строкам
            delete_orig: bool (default False) - флаг на удаление смещаемой фичи (оригинальной)
            colname_for_lag: str (default Добыча воды за 2 ч |м3) - название фичи для сдвига
        """
        print(f"Смещение признака '{colname_for_lag}' на lag {lag}") if self.verbose else None
        if lag:
            self.df = pd.concat([self.df, 
                                self.df[colname_for_lag].shift(lag).
                                rename(f"{colname_for_lag} лаг:({lag})")], axis=1).dropna().reset_index(drop=True)
            if delete_orig:
                self.df.drop(columns=colname_for_lag, inplace=True)
            
            print(f"'{colname_for_lag}' смещено на {lag}") if self.verbose else None
        else:
            print('Смещение установлено в 0. Пропускаем...') if self.verbose else None

    def filter_by_hours(self, hours: list=[8, 20]):
        """
                Фильтрует датасет по `hours` часам
                Необходима фича 'Час'
                *Метод parse_datetime достает фичу 'Час'*
            hours: list (default [8, 20]) - int значения часов для фильтрации
        """
        print(f"Фильтрация по {hours} часам") if self.verbose else None
        if 'Час' in self.cols:
            self.df = self.df[self.df['Час'].isin(hours)].reset_index(drop=True)
            print(self.df['Час'].value_counts()) if self.verbose else None
        else:
            print("Признак 'Час' не найден")

    def smooth(self, columns, frac=0.001, window=5, polyorder=3, insert=True, method='lowess'):
        """
            Сглаживание значений в указанных признаках
            Поддерживаемые методы:
            - lowess - Сглаживание по методу Лоусса
            - rolling - Скользящее среднее окно
            - savgol_filter (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) - Фильтр Савицкого-Голея 
                - подгоняет последующие окна смежных данных с полиномом низкого порядка
        """
        columns = self.columns(columns)
        print("Восстановление выбросов...") if self.verbose else None
        for idx, col in enumerate(columns):
            match method:
                case 'lowess':
                    smoothed = lowess(self.df[col], range(len(self.df)), frac=frac)
                case 'rolling':
                    smoothed = self.df[col].rolling(window=window).mean()
                case 'savgol_filter':
                    smoothed = signal.savgol_filter(self.df[col],
                               window, # window size used for filtering
                               polyorder) # order of fitted polynomial
                case _:
                    raise ValueError("Указан неподдерживаемый метод")

            if insert:
                index = self.cols.get_loc(col) + 1
                new_col = f'smoothed|{col}|{method}'
                self.df.insert(index, new_col, smoothed)
            else:
                self.df.loc[:, col] = smoothed

    def show_scatter(self, column):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df['YY-MM-DD HH:00'],
            y=self.df[column],
            marker=dict(size=2, color='black',),
            opacity=0.25,
            name=column
        ))

        fig.add_trace(go.Scatter(
            x=self.df['YY-MM-DD HH:00'],
            y=self.df["smoothed|"+column],
            marker=dict(
                size=6,
                color='royalblue',
                symbol='circle-open'
            ),
            name='Smoothed'
        ))

        fig.add_trace(go.Scatter(
            x=self.df['YY-MM-DD HH:00'],
            y=self.df["smoothed|"+column],
            mode='markers',
            marker=dict(
                size=6,
                color='mediumpurple',
                symbol='triangle-up'
            ),
            name='Smoothed scatter'
        ))

        fig.update_layout(height=800)
        fig.show()


    def convert_datetime(self, drop=False):
        datetimes = ['Год', 'Месяц','День','Час']
        self.df.insert(0, 'YY-MM-DD HH:00', self.df[datetimes].apply(
            lambda x: f'{x.iloc[0]:04}-{x.iloc[1]:02}-{x.iloc[2]:02} {x.iloc[3]:02}:00', 
            axis = 1)) 
        self.df.sort_values('YY-MM-DD HH:00', inplace=True)
        if drop:
            self.df.drop(columns=datetimes, inplace=True)
    
    def save(self, filename='./data/preprocessed/filled_outliers', extension='csv'):
        match extension:
            case 'csv':
                self.df.to_csv(f'{filename}.csv', index=False)

    def new_feature(self, columns, newcol, agg_f:str='max', drop=False):
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
           

    def drop(self, columns):
        columns = self.columns(columns)
        self.df.drop(columns, axis=1, inplace=True)

    def time_series(self, columns=None, appendix_cols: list=['Добыча воды за 2 ч |м3 лаг:(-1)'], 
                show=True, figsize=(1000, 3000), log=(True,False), save_path=None):
        columns = self.columns(columns)
        if appendix_cols:
            columns += appendix_cols
        if 'YY-MM-DD HH:00' not in self.cols:
            columns.append('YY-MM-DD HH:00')
        print("График отображает признаки: ", columns) if self.verbose else None
    
        try:
            plot = px.line(self.df, x='YY-MM-DD HH:00', y=columns,
                        height=figsize[0], width=figsize[1], log_x=log[0], log_y=log[1])
            if save_path:
                plot.write_html(save_path)
                print("Файл сгенерирован.") if self.verbose else None
            if show:
                plot.show()
        except BaseException as err:
            print(err)

    
    def __iter__(self):
        return self.cols.__iter__()
    
    def report(self, columns=None, filepath='./reports/report.html'):
        columns = self.columns(columns)
        ProfileReport(self.df[columns], 
                      title="Profiling Report").to_file(output_file=filepath)
        
    def scatter_matrix(self, columns=None, filepath='./plots/scatter_matrix.html', size=None, show=False):
        columns = self.columns(columns)
        fig = px.scatter_matrix(self.df[columns], height=size[0], width=size[1])
        fig.update_traces(diagonal_visible=False, showlowerhalf=False)
        fig.write_html(filepath)
        if show:
            fig.show()

    def corr_matrix(self, columns=None, filepath='./plots/corr_matrix.html', method='pearson', size=None, show=False, textfont_size=10):
        columns = self.columns(columns)
        corr = self.df[columns].corr(method)
    
        fig = px.imshow(corr, text_auto=True, height=size[0], width=size[1],  color_continuous_scale=sns_colormap)
        fig.update_traces(textfont_size=textfont_size,  texttemplate = "%{z:.2f}")
        fig.write_html(filepath)
        if show:
            fig.show()

    @property
    def cols(self):
        return self.df.columns

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



# slice функция по индексу для словарей
slice = lambda d, start=0, stop=None, step=1: dict(itertools.islice(d.items(), start, stop, step))

