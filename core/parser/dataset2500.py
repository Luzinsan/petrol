import pandas

from typing import List

from base import Base


class Dataset2500(Base):
    def __init__(self, filename: str):
          super().__init__(filename)

    def get_muis_2500(self, df, month, is_indicators=False) -> pandas.DataFrame:
        # Найти индекс строки с нужным значением в столбце
        if month == 1 and is_indicators:
            mask = (df['Режимный лист МУИС-2500 Воронцовского месторождения'] == 'Дата/Время мск') \
                & (df['Режимный лист МУИС-2500 Воронцовского месторождения'].shift(1) != 'Режимный лист МУИС-1250 Воронцовского месторождения') \
                & ((df['Unnamed: 2'] == 'Блок манифольда') | (df['Unnamed: 3'] == 'Блок манифольда') & (df['Unnamed: 2'] != 'Блок манифольда'))
        else:
            mask = (df['Режимный лист МУИС-2500 Воронцовского месторождения'] == 'Дата/Время мск') \
                & (df['Режимный лист МУИС-2500 Воронцовского месторождения'].shift(1) != 'Режимный лист МУИС-1250 Воронцовского месторождения') \
                & (df['Unnamed: 2'] == 'Блок манифольда')


        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        # Сохранить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]:index_to_keep[0]+16]
        index_to_keep = index_to_keep.delete(0)
        df_new = pandas.concat([df.iloc[idx+1:idx+16] for idx in index_to_keep])
        frames = [df_first_rows, df_new]

        result = pandas.concat(frames)
        result = result.iloc[1: , :]

        return result

    
    def get_fresh_water(self, df: pandas.DataFrame, column: str) -> pandas.DataFrame:
        mask = (df[column] == 'Расход пресной воды')

        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        # Сохранить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]+2:index_to_keep[0]+15]
        index_to_keep = index_to_keep.delete(0)
        df_new = pandas.concat([df.iloc[idx+3:idx+15] for idx in index_to_keep])
        frames = [df_first_rows, df_new]

        result = pandas.concat(frames)
        result = result.iloc[1: , :]

        return result


    def get_salt(self, df: pandas.DataFrame, column_time: str) -> pandas.DataFrame:
        first_index = 2
        shift = 29
        
        df_first_rows = df.iloc[first_index:shift]

        mask = (df[column_time] == 'Время')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index

        # Сохранить строки, начиная с найденного индекса
        df_new = pandas.concat([df.iloc[idx+3:idx+3+shift] for idx in index_to_keep])
        frames = [df_first_rows, df_new]

        result = pandas.concat(frames)
        result = result.iloc[1: , :]

        return result


    def rename_cols(self, df: pandas.DataFrame, df_type: str) -> pandas.DataFrame:
        if df_type == 'muis_2500':
            df.columns = [
                'Дата',  'Блок манифольда|P бм  |кгс/см²',
                'Блок манифольда|t жидкости| °С', 'С-1|P сеп. |кгс/см²',
                'С-1|t жидкости |°С', 'С-1|L жидкости| см', 'С-2/1|P сеп. |кгс/см²',
                'С-2/1|t жидкости|°С', 'С-2/1|L межфазный|см', 'С-2/1|L нефти|см',
                'С-2/2|P сеп. |кгс/см²', 'С-2/2|t жидкости|°С', 'С-2/2|L межфазный|см',
                'С-2/2|L нефти|см', 'ОН-1/1|P отс. |кгс/см²', 'ОН-1/1|t жидкости|°С',
                'ОН-1/1|L межфазный|см', 'ОН-1/2|P отс. |кгс/см²',
                'ОН-1/2|t жидкости|°С', 'ОН-1/2|L  межфазный|см', 'С-3|t жидкости|°С',
                'С-3|L нефти|см', 'П-1|Т нефти на входе|°С', 'П-1|Т нефти на выходе|°С',
                'П-1|Р нефти на входе|кгс/см²', 'П-1|Р нефти на выходе|кгс/см²',
                'П-1|Т теплоно-сителя|°С', 'П-1|Т дымовых газов|°С',
                'П-1|Р на горелку|кгс/см2', 'П-2|Т нефти на входе|°С',
                'П-2|Т нефти на выходе|°С', 'П-2|Р нефти на входе|кгс/см²',
                'П-2|Р нефти на выходе|кгс/см²', 'П-2|Т теплоно-сителя|°С',
                'П-2|Т дымовых газов|°С', 'П-2|Р на горелку|кгс/см3',
                'П-3|Т нефти на входе|°С', 'П-3|Т нефти на выходе|°С',
                'П-3|Р нефти на входе|кгс/см²', 'П-3|Р нефти на выходе|кгс/см²',
                'П-3|Т теплоно-сителя|°С', 'П-3|Т дымовых газов|°С',
                'П-3|Р на горелку|кгс/см4'
            ]

        elif df_type == 'water_2500':
            df.columns = [
                'Дата', 'Unnamed: 2',
                'Добыча воды за 2 ч',
                'Добыча воды за 2 ч, расч',
                'Добыча воды за 2 ч, расч, по ТОРам']

        elif df_type == 'fresh_water_2500':
            df.columns = ['Дата','Расход пресной воды']

        return df


    def clear_dataset_2500_indicators(self, df: pandas.DataFrame, i: int) -> pandas.DataFrame:
        df_indicator = df.copy()
        df_indicator = df_indicator.iloc[:, 0: 45]
        df_indicator = df_indicator.drop(['Unnamed: 1', 'Unnamed: 23'], axis=1)
        df_indicator = self.get_muis_2500(df_indicator, i, True)
        df_indicator = self.rename_cols(df_indicator, 'muis_2500')
        df_indicator = df_indicator.reset_index()
        df_indicator = df_indicator.drop(['index'], axis=1)
        df_indicator = self.fill_date(df_indicator)
        # df_indicator = self.drop_each_non_exist_hour(df_indicator)

        return df_indicator


    def clear_dataset_2500_water_hourly(self, df: pandas.DataFrame, i: int) -> pandas.DataFrame:
        df_water_hourly = df.copy()
        df_water_hourly = df_water_hourly.iloc[:, [0, 2, 57, 59, 60]]
        df_water_hourly = self.get_muis_2500(df_water_hourly, i)
        df_water_hourly = df_water_hourly.reset_index()
        df_water_hourly = df_water_hourly.drop(['index'], axis=1)
        df_water_hourly = self.rename_cols(df_water_hourly, 'water_2500')
        # df_water_hourly = self.drop_each_non_exist_hour(df_water_hourly)
        df_water_hourly = df_water_hourly.drop(['Дата'], axis=1)

        return df_water_hourly

    
    def clear_dataset_fresh_water(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_fresh_water = df.copy()
        df_fresh_water = df_fresh_water.iloc[:, [0, 51]]

        df_fresh_water = self.get_fresh_water(df_fresh_water, 'Unnamed: 51')
        df_fresh_water = df_fresh_water.reset_index()
        df_fresh_water = df_fresh_water.drop(['index'], axis=1)
        df_fresh_water = self.rename_cols(df_fresh_water, 'fresh_water_2500')
        df_fresh_water = self.fill_date(df_fresh_water)

        return df_fresh_water

    
    def clear_dataset_salt(self, df: pandas.DataFrame) -> pandas.DataFrame:
        temp = df.iloc[:, [0, 71, 72, 74]]
        res = self.get_salt(temp, 'Время')
        res = res.reset_index(drop=True)

        res['new_date'] = pandas.to_datetime(res['Режимный лист МУИС-2500 Воронцовского месторождения'], errors='coerce')
        result_df = res[res['new_date'].notna()]
        df_unique = result_df.drop_duplicates(subset='new_date', keep='first')

        # создание колонки, заполненной подряд идущими датами
        for idx in list(df_unique.index):
            # Получаем дату по текущему индексу
            current_date = res.loc[idx, 'new_date']  # Предполагаем, что столбец с датами называется 'date'

            # Присваиваем эту дату следующим 28 записям
            res.loc[idx:idx + 28, 'prolonged_date'] = current_date

        res['prolonged_date'] = pandas.to_datetime(res['prolonged_date'], format='%d.%m.%Y')
        res['Время'] = res['Время'].apply(self.replace_with_time)
        res['Время'] = pandas.to_datetime(res['Время'], format='%H:%M:%S').dt.time

        # Объединение колонки 'date' и 'time' в новую колонку 'datetime'
        res['Дата'] = res['prolonged_date'] + pandas.to_timedelta(res['Время'].astype(str))

        res = res.drop(['Режимный лист МУИС-2500 Воронцовского месторождения', 'Время', 'new_date', 'prolonged_date'], axis = 1)
        res = res.dropna(subset = 'Сод. Cl соед., мг/дм3')
        res = res.reset_index(drop=True)

        return res


    def create_dataset(self, parsing_pages: List[str]) -> pandas.DataFrame:

        final_dataset = pandas.DataFrame()
        for i, page in enumerate(parsing_pages):
            df_month = self.xl.parse(page)
            indicators = self.clear_dataset_2500_indicators(df_month, i)
            water_hourly = self.clear_dataset_2500_water_hourly(df_month, i)
            
            result = pandas.concat([indicators, water_hourly], axis=1)
            if i == 0:
                final_dataset = result.copy()
            else:
                final_dataset = pandas.concat([final_dataset, result]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

            final_dataset = self.check_dataset(final_dataset)
            final_dataset.to_csv(f'data/preprocessed/{self.filename}_2500_clear.csv')

        return final_dataset

    
    def create_dataset_fresh_water(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()

        for i, page in enumerate(parsing_pages):
            df_month = self.xl.parse(page)
            fresh_water = self.clear_dataset_fresh_water(df_month)
            result = fresh_water
            print(f'Размерность пресной воды: {fresh_water.shape} - {page}')
            if i == 0:
                final_dataset = result.copy()
            else:
                final_dataset = pandas.concat([final_dataset, result]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

        final_dataset = self.check_dataset(final_dataset)
        final_dataset.to_csv(f'data/preprocessed/{self.filename}_2500_fresh_water_clear.csv')

        return final_dataset


    def create_dataset_salt(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()

        for i, page in enumerate(parsing_pages):
            df_month = self.xl.parse(page)
            salt = self.clear_dataset_salt(df_month)
            result = salt
            print(f'Размерность пресной воды: {salt.shape} - {page}')
            if i == 0:
                final_dataset = result.copy()
            else:
                final_dataset = pandas.concat([final_dataset, result]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

        final_dataset = self.check_dataset(final_dataset)
        final_dataset.to_csv(f'data/preprocessed/{self.filename}_2500_salt_clear.csv')

        return final_dataset
    
dataset = Dataset2500('data/raw/Режимный лист МУИС 2023.xlsx')

parsing_pages = [ 
                'Январь 2023',
                  'Февраль 2023',
                  'Март 2023',
                #   'Апрель 2022',
                #   'Май 2022',
                #   'Июнь 2022',
                #   'Июль 2022',
                #   'Август 2022',
                #   'Сентябрь 2022',
                #   'Октябрь 2022',
                #   'Ноябрь 2022',
                #   'Декабрь 2022'
                ]

data_full = dataset.create_dataset_salt(parsing_pages)
data_salt = dataset.create_dataset(parsing_pages)
data_moisture = dataset.create_dataset_fresh_water(parsing_pages)