import pandas as pd
import os
import numpy as np
import itertools
import datetime


class Dataset:

    def __init__(self, path, dropna=True):
        print("Чтение датасета...")
        self.df: pd.DataFrame = pd.read_excel(path, parse_dates=['Дата'])
        self.original = self.df.copy()
        if dropna:
            print(f"Удаленые записи\n{self.df.isna().sum()}")
            self.df = self.df.dropna().reset_index(drop=True)
        print("Завершено успешно")

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
        print("Парсинг Даты и Времени...")
        if parse_date:
            loc=self.df.columns.get_loc('Дата')
            self.df.insert(loc, 'День', self.df['Дата'].dt.day)
            self.df.insert(loc, 'Месяц', self.df['Дата'].dt.month)
            self.df.insert(loc, 'Год', self.df['Дата'].dt.year)
            
        if parse_time:
            self.df.insert(self.df.columns.get_loc('Время'), 'Час', self.df['Время'].apply(lambda x: x.hour))
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
        print(f"Смещение признака '{colname_for_lag}' на lag {lag}")
        if lag:
            self.df = pd.concat([self.df, 
                                self.df[colname_for_lag].shift(lag).
                                rename(f"{colname_for_lag} лаг:({lag})")], axis=1).dropna().reset_index(drop=True)
            if delete_orig:
                self.df.drop(columns=colname_for_lag, inplace=True)
            
            print(f"'{colname_for_lag}' смещено на {lag}")
        else:
            print('Смещение установлено в 0. Пропускаем...')

    def filter_by_hours(self, hours: list=[8, 20]):
        """
                Фильтрует датасет по `hours` часам
                Необходима фича 'Час'
                *Метод parse_datetime достает фичу 'Час'*
            hours: list (default [8, 20]) - int значения часов для фильтрации
        """
        print(f"Фильтрация по {hours} часам")
        if 'Час' in self.df.columns:
            self.df = self.df[self.df['Час'].isin(hours)].reset_index(drop=True)
            print(self.df['Час'].value_counts())
        else:
            print("Признак 'Час' не найден")

    def recovery_outliers(self, columns, quantiles, threshold, threshold_for_frequency=20, insert=True):
        """

        """
        print("Восстановление выбросов...")
        for idx, col in enumerate(columns):
            outliers = tuple(self.df[col].quantile(quantiles[idx]).values)
            print(f"{col}. Значения квантилей {quantiles[idx]}: {outliers}")
        
            value_counts = self.df[col].value_counts()
            value_counts: pd.Series  = value_counts[value_counts<threshold[idx]]
            
            print("Кол-во редких наблюдений по частоте встречаемости:", value_counts.values.sum())
            try:
                rare_values = list(value_counts.keys())[-threshold_for_frequency:]
                
            except BaseException:
                rare_values = list(value_counts.keys())
            print("Учитываются значения: ", rare_values)
        
            mask = (
                    (~self.df[col].between(*outliers) & self.df[col].isin(rare_values))
                #   | ( self.df[col]<=0.0)
                )
            
            if insert:
                index = self.df.columns.get_loc(col) + 1
                new_col = "outlier|"+col
                self.df.insert(index, new_col, 
                        np.where(mask, np.nan, self.df[col]))
                col = new_col
            else:
                self.df.loc[mask, col] = np.nan
            self.df[col] = self.df[col].ffill()


# slice функция по индексу для словарей
slice = lambda d, start=0, stop=None, step=1: dict(itertools.islice(d.items(), start, stop, step))




# DATA = './data'
# df = Dataset(os.path.join(DATA, 'processed.xlsx'))
# df.parse_datetime()
# df.set_lag(lag=-1)
# df.set_lag(lag=1)
# cols_for_delete = [
    
#        # оставляем пока что П-_|Т нефти на входе|°С
#        #  'П-1|Т нефти на входе|°С',
#        #  'П-2|Т нефти на входе|°С',
#        #  'П-3|Т нефти на входе|°С',
#        # median аггрегация => П-123median|Т нефти на выходе|°С
#        #  'П-1|Т нефти на выходе|°С',
#        #  'П-2|Т нефти на выходе|°С',
#        #  'П-3|Т нефти на выходе|°С',
#        # max аггрегация => П-123max|Р нефти на входе|кгс/см²
#        #  'П-1|Р нефти на входе|кгс/см²',
#        #  'П-2|Р нефти на входе|кгс/см²',
#        #  'П-3|Р нефти на входе|кгс/см²',
#        # max аггрегация => П-123max|Р нефти на выходе|кгс/см²
#        #  'П-1|Р нефти на выходе|кгс/см²',
#        #  'П-2|Р нефти на выходе|кгс/см²',
#        #  'П-3|Р нефти на выходе|кгс/см²',
#        # константные значения
#        #  'П-1|Р на горелку|кгс/см2',
#        #  'П-2|Р на горелку|кгс/см3',
#        #  'П-3|Р на горелку|кгс/см4',
#        # БЕВ'ы 
#        'БЕВ-1|L воды|см', 'БЕВ-1|V  воды|м3',
#        'БЕВ-1|V нефти|м3', 'БЕВ-1|t воды|°С', 'БЕВ-2|L воды|см',
#        'БЕВ-2|V  воды|м3', 'БЕВ-2|V нефти|м3', 'БЕВ-2|t воды|°С',
#        'БЕВ-3|L воды|см', 'БЕВ-3|V  воды|м3', 'БЕВ-3|V нефти|м3',
#        'БЕВ-3|t воды|°С',
#        'Добыча воды за сутки|м3',
#        ]

# df.df.drop(cols_for_delete, axis=1, inplace=True)

# columns = df.df.columns[4:]
# quantiles = [[0.001, 0.999]]*len(columns)
# threshold = [3]*len(columns)
# df.recovery_outliers(columns, quantiles=quantiles, threshold=threshold)
# df.df.to_csv("df_with_lag.csv",index=False)




