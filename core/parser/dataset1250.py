import pandas

from typing import List

from base import Base


class Dataset1250(Base):
    def __init__(self, filename):
        super().__init__(filename)


    def get_muis_1250(self, df: pandas.DataFrame) -> pandas.DataFrame:
        mask = (df['Режимный лист МУИС-2500 Воронцовского месторождения'] == 'Режимный лист МУИС-1250 Воронцовского месторождения') 

        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Удалить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]:index_to_keep[0]+16]
        index_to_keep = index_to_keep.delete(0)
        df_new = pandas.concat([df.iloc[idx+1:idx+16] for idx in index_to_keep])
        frames = [df_first_rows, df_new]

        result = pandas.concat(frames)
        result = result.iloc[1: , :]
        result.dropna(subset=['Режимный лист МУИС-2500 Воронцовского месторождения'], inplace=True)

        return result


    def get_moisture(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # Unnamed: 31 - для Марта 2023
        mask = (df['Unnamed: 27'] == 'Показания влагомера') 
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Удалить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]+2:index_to_keep[0]+15]
        index_to_keep = index_to_keep.delete(0)
        df_new = pandas.concat([df.iloc[idx+3:idx+15] for idx in index_to_keep])
        frames = [df_first_rows, df_new]

        result = pandas.concat(frames)
        result = result.iloc[1: , :]

        return result

  
    def get_fresh_water(self, df: pandas.DataFrame) -> pandas.DataFrame:
        mask = (df['Unnamed: 36'] == 'Расход пресной воды МУИС 1250')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Срез строк
        df_first_rows = df.iloc[index_to_keep[0]+2:index_to_keep[0]+16]
        index_to_keep = index_to_keep.delete(0)
        df_new = pandas.concat([df.iloc[idx+3:idx+16] for idx in index_to_keep])
        frames = [df_first_rows, df_new]

        result = pandas.concat(frames)
        result = result.iloc[1: , :]

        return result

    
    def get_salt(self, df, column_time: str) -> pandas.DataFrame:
        shift = 29

        mask = (df[column_time] == 'Время')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index

        # Удалить строки, начиная с найденного индекса
        df_new = pandas.concat([df.iloc[idx+4:idx+4+shift] for idx in index_to_keep])

        return df_new


    def rename_cols(self, df: pandas.DataFrame, df_type: str) -> pandas.DataFrame:
        if df_type == 'muis_1250':
            df.columns = [
                'Дата',  'Блок манифольда|P бм  |кгс/см²',
                'Блок манифольда|t жидкости| °С', 'С-1|P сеп. |кгс/см²',
                'С-1|t жидкости |°С', 'С-1|L жидкости| см', 'С-2|P сеп. |кгс/см²',
                'С-2|t жидкости|°С', 'С-2|L межфазный|см', 'С-2|L нефти|см',
                'ОН-1|t жидкости|°С',
                'ОН-1|L межфазный|см', 'ОН-1|P отс. |кгс/см²',
                'С-3|t жидкости|°С',
                'С-3|L нефти|см', 'П-1|Т нефти на входе|°С', 'П-1|Т нефти на выходе|°С',
                'П-1|Р нефти на входе|кгс/см²', 'П-1|Р нефти на выходе|кгс/см²',
                'П-1|Т теплоно-сителя|°С', 'П-1|Т дымовых газов|°С',
                'П-1|Р на горелку|кгс/см2', 'П-2|Т нефти на входе|°С',
                'П-2|Т нефти на выходе|°С', 'П-2|Р нефти на входе|кгс/см²',
                'П-2|Р нефти на выходе|кгс/см²', 'П-2|Т теплоно-сителя|°С',
                'П-2|Т дымовых газов|°С', 'П-2|Р на горелку|кгс/см3',
                'БЕВ-1|L воды|см', 'БЕВ-1|V воды|м3',
                'БЕВ-1|V нефти', 'БЕВ-1|t воды'
            ]

        elif df_type == 'moisture':
            df.columns = [
                'Дата',
                'Добыча воды за 2ч',
                'Показания влагомера',
            ]

        elif df_type == 'fresh_water':
            df.columns = [
                'Расход пресной воды'
            ]

        elif df_type == 'salt':
            df.columns = [
                'Место отбора',
                'Сод. Cl соед., мг/дм3',
                'Дата'
            ]

        return df


    def clear_dataset_moisture(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_moisture = df.copy()
        # Для Марта 2023 - 0, 26, 31
        df_moisture = df_moisture.iloc[:, [0, 22, 27]]
        df_moisture = self.get_moisture(df_moisture)
        df_moisture = self.rename_cols(df_moisture, 'moisture')
        df_moisture = df_moisture.reset_index(drop=True)
        df_moisture = self.fill_date(df_moisture)

        return df_moisture


    def clear_dataset_1250_indicators(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_indicator_1250 = df.copy()
        df_indicator_1250 = df_indicator_1250.iloc[:, 0: 35]
        df_indicator_1250 = df_indicator_1250.drop(['Unnamed: 1', 'Unnamed: 16'], axis=1)
        df_indicator_1250 = self.get_muis_1250(df_indicator_1250)
        df_indicator_1250 = self.rename_cols(df_indicator_1250, 'muis_1250')
        df_indicator_1250 = df_indicator_1250.reset_index()
        df_indicator_1250 = df_indicator_1250.drop(['index'], axis=1)
        df_indicator_1250 = self.fill_date(df_indicator_1250)
        # df_indicator_1250 = self.drop_each_non_exist_hour(df_indicator_1250)
        df_indicator_1250 = df_indicator_1250.dropna()

        return df_indicator_1250


    def clear_dataset_1250_fresh_water(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_fresh_water = df.iloc[:, [36]]
        df_fresh_water = self.get_fresh_water(df_fresh_water)
        df_fresh_water = df_fresh_water.reset_index()
        df_fresh_water = df_fresh_water.drop(['index'], axis=1)
        df_fresh_water = df_fresh_water.dropna()
        df_fresh_water = self.rename_cols(df_fresh_water, 'fresh_water')

        return df_fresh_water

    
    def clear_dataset_salt(self, df: pandas.DataFrame, time_col: str, salt_col: str) -> pandas.DataFrame:
        temp = df.iloc[:, [0, 48, 49, 51]]
        res = self.get_salt(temp, time_col)
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
        res[time_col] = res[time_col].apply(self.replace_with_time)
        res[time_col] = pandas.to_datetime(res[time_col], format='%H:%M:%S').dt.time

        # Объединение колонки 'date' и 'time' в новую колонку 'datetime'
        res['Дата'] = res['prolonged_date'] + pandas.to_timedelta(res[time_col].astype(str))

        res = res.drop(['Режимный лист МУИС-2500 Воронцовского месторождения', time_col, 'new_date', 'prolonged_date'], axis = 1)
        res = res.dropna(subset = salt_col)
        res = res.reset_index(drop=True)

        res = self.rename_cols(res, 'salt')

        return res
      

    def create_dataset(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()
        for i, page in enumerate(parsing_pages):
            df_month = self.xl.parse(page)
            indicators = self.clear_dataset_1250_indicators(df_month)
            fresh_water = self.clear_dataset_1250_fresh_water(df_month)
            result = pandas.concat([indicators, fresh_water], axis=1)

            if i == 0:
                final_dataset = result.copy()
            else:
                final_dataset = pandas.concat([final_dataset, result]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

            final_dataset = self.check_dataset(final_dataset)
            final_dataset.to_csv(f'data/preprocessed/{self.filename}_1250_clear.csv')

        return final_dataset


    def create_dataset_1250_moisture(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()
        for i, page in enumerate(parsing_pages):
            df_month = self.xl.parse(page)
            moisture = self.clear_dataset_moisture(df_month)
            if i == 0:
                final_dataset = moisture.copy()
            else:
                final_dataset = pandas.concat([final_dataset, moisture]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

            final_dataset = self.check_dataset(final_dataset)
            final_dataset = final_dataset.dropna()
            final_dataset.to_csv(f'data/preprocessed/{self.filename}_1250_moisture_clear.csv')

        return final_dataset


    def create_dataset_salt(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()

        for i, page in enumerate(parsing_pages):
            df_month = self.xl.parse(page)
            salt = self.clear_dataset_salt(df_month, 'Unnamed: 48', 'Unnamed: 51')
            result = salt
            print(f'Размерность соли: {salt.shape} - {page}')
            if i == 0:
                final_dataset = result.copy()
            else:
                final_dataset = pandas.concat([final_dataset, result]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

        final_dataset = self.check_dataset(final_dataset)
        final_dataset.to_csv(f'data/preprocessed/{self.filename}_1250_salt_clear.csv')

        return final_dataset
    

dataset = Dataset1250('data/raw/Режимный лист МУИС 2023.xlsx')

parsing_pages = [ 
                'Январь 2023',
                  'Февраль 2023',
                #   'Март 2023',
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
data_moisture = dataset.create_dataset_1250_moisture(parsing_pages)