from typing import List 

import pandas

from base import Base


class Dataset4000Line2(Base):
    def __init__(self, filename):
        super().__init__(filename)


    def get_muis_4000_line_2(self, df: pandas.DataFrame) -> pandas.DataFrame:
        mask = (df['        Режимный лист МУИС-4000 техн. линия № 1 Воронцовского месторождения'] == 'Режимный лист МУИС-4000 техн. линия № 2 Воронцовского месторождения')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Строки, начиная с найденного индекса
        df_line_one = df.iloc[index_to_keep[0]+2:index_to_keep[0]+16]
        result = df_line_one.iloc[:, 0:37]
        result.dropna(subset=['        Режимный лист МУИС-4000 техн. линия № 1 Воронцовского месторождения'], inplace=True)

        return result


    def get_moisture_4000_line_2(self, df: pandas.DataFrame, column: str) -> pandas.DataFrame:
        mask = (df[column] == 'Показания влагомера')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]+2:index_to_keep[0]+15]
        result = df_first_rows.iloc[1: , :]

        return result

    def get_fresh_water_4000_line_2(self, df: pandas.DataFrame, column: str) -> pandas.DataFrame:
        mask = (df[column] == 'Расход пресной воды')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Сохранить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]+3:index_to_keep[0]+15]
        result = df_first_rows

        return result


    def get_salt_4000_line_2(self, df: pandas.DataFrame, column_time: str, column_salt: str) -> pandas.DataFrame:
        mask = (df[column_time] == 'Время') & (df[column_salt] == 'Сод. Cl соед., мг/дм3')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index

        mask_last_idx = (df[column_time] == 'ТОР-2')
        last_index_to_keep = df[mask_last_idx]
        last_index_to_keep = last_index_to_keep.index
        print(last_index_to_keep)
        # Удалить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]+4:last_index_to_keep[0]]
        result = df_first_rows

        return result


    def rename_cols(self, df: pandas.DataFrame, df_type: str) -> pandas.DataFrame:
        if df_type == 'muis_4000_line_2':
            df.columns = [
                'Дата',  'Блок манифольда|P бм  |кгс/см²',
                'Блок манифольда|t жидкости| °С',
                'С-1/3|P сеп. |кгс/см²', 'С-1/3|t жидкости |°С', 'С-1/3|L жидкости| см',
                'С-2/3|P сеп. |кгс/см²', 'С-2/3|t жидкости |°С', 'С-2/3|L жидкости| см', 'С-2/3|L нефти',
                'ОН-1/3|P отс. |кгс/см²', 'ОН-1/3|t жидкости|°С', 'ОН-1/3|L межфазный|см',
                'С-3/3|t жидкости|°С', 'С-3/3|L нефти|см',
                'ЭДГ-1/3|L меж. Фаз.', 'ЭДГ-1/3|Р', 'ЭДГ-1/3|Т', 'ЭДГ-1/3|Сила тока', 'ЭДГ-1/3|Напряжение',
                'П-4|Т нефти на входе|°С', 'П-4|Т нефти на выходе|°С',
                'П-4|Р нефти на входе|кгс/см²', 'П-4|Р нефти на выходе|кгс/см²',
                'П-4|Т теплоно-сителя|°С', 'П-4|Т дымовых газов|°С',
                'П-4|Р на горелку|кгс/см2', 'П-5|Т нефти на входе|°С',
                'П-5|Т нефти на выходе|°С', 'П-5|Р нефти на входе|кгс/см²',
                'П-5|Р нефти на выходе|кгс/см²', 'П-5|Т теплоно-сителя|°С',
                'П-5|Т дымовых газов|°С', 'П-5|Р на горелку|кгс/см3',

            ]

        elif df_type == 'moisture_4000_line_2':
            df.columns = [
                'Дата',
                'Показания влагомера',
            ]

        elif df_type == 'fresh_water_4000_line_2':
            df.columns = [
                'Расход пресной воды'
            ]

        elif df_type == 'salt_4000_line_2':
            df.columns = [
                'Дата', 'Место сбора', 'Сод. Cl соед., мг/дм3'
            ]

        return df


    def clear_dataset_4000_line_2_indicators(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_indicator_4000 = df.copy()
        df_indicator_4000 = df_indicator_4000.iloc[:, 0: 35]
        df_indicator_4000 = df_indicator_4000.drop(['Unnamed: 1'], axis=1)
        df_indicator_4000 = self.get_muis_4000_line_2(df_indicator_4000)
        df_indicator_4000 = self.rename_cols(df_indicator_4000, 'muis_4000_line_2')
        df_indicator_4000 = df_indicator_4000.reset_index()
        df_indicator_4000 = df_indicator_4000.drop(['index'], axis=1)
        df_indicator_4000 = self.fill_date(df_indicator_4000)
        # df_indicator_4000 = self.drop_each_non_exist_hour(df_indicator_4000)
        df_indicator_4000 = df_indicator_4000.dropna()

        return df_indicator_4000



    def clear_dataset_moisture_4000_line_2(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_moisture = df.iloc[:, [0, 35]]
        df_moisture = self.get_moisture_4000_line_2(df_moisture, 'Unnamed: 35')
        df_moisture = df_moisture.dropna()
        df_moisture = self.rename_cols(df_moisture, 'moisture_4000_line_2')
        df_moisture = df_moisture.reset_index(drop=True)
        df_moisture = self.fill_date(df_moisture)
        # df_moisture = self.drop_each_non_exist_hour(df_moisture)
        df_moisture = df_moisture.drop(['Дата'], axis=1)

        return df_moisture


    def clear_dataset_4000_line_2_fresh_water(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_fresh_water = df.copy()
        df_fresh_water = df_fresh_water.iloc[:, [40]]

        df_fresh_water = self.get_fresh_water_4000_line_2(df_fresh_water, 'Unnamed: 40')
        df_fresh_water = self.rename_cols(df_fresh_water, 'fresh_water_4000_line_2')
        df_fresh_water = df_fresh_water.reset_index()
        df_fresh_water = df_fresh_water.drop(['index'], axis=1)
        df_fresh_water = df_fresh_water.dropna()

        return df_fresh_water


    def clear_dataset_4000_line_2_salt(self, df: pandas.DataFrame, pages: List[str], i: int, time_column: str) -> pandas.DataFrame:
        df_salt = df.iloc[:, [52, 53, 55]]
        df_salt = self.get_salt_4000_line_2(df_salt, 'Unnamed: 52', 'Unnamed: 55')
        df_salt = df_salt.dropna()
        date: str = pages[i]

        df_salt['Дата'] = date
        df_salt['Дата'] = pandas.to_datetime(df_salt['Дата'], format='%d.%m.%Y')
        df_salt[time_column] = pandas.to_datetime(df_salt[time_column], format='%H:%M:%S').dt.time

        # Объединение колонки 'date' и 'time' в новую колонку 'datetime'
        df_salt[time_column] = df_salt['Дата'] + pandas.to_timedelta(df_salt[time_column].astype(str))
        df_salt = df_salt.drop(['Дата'], axis=1)
        df_salt = self.rename_cols(df_salt, 'salt_4000_line_2')

        return df_salt


    def create_dataset(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()
        for i, page in enumerate(parsing_pages):
            df_day = self.xl.parse(page)
            indicators = self.clear_dataset_4000_line_2_indicators(df_day)
            moisture = self.clear_dataset_moisture_4000_line_2(df_day)
            fresh_water = self.clear_dataset_4000_line_2_fresh_water(df_day)
            result = pandas.concat([indicators, moisture, fresh_water], axis=1)
            if i == 0:
                final_dataset = result.copy()
            else:
                final_dataset = pandas.concat([final_dataset, result]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

            final_dataset = self.check_dataset(final_dataset)
            final_dataset.to_csv(f'data/preprocessed/{self.filename}_clear_Line2.csv')

        return final_dataset


    def create_dataset_salt(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()
        for i, page in enumerate(parsing_pages):
            df_day = self.xl.parse(page)
            salt = self.clear_dataset_4000_line_2_salt(df_day, parsing_pages, i, 'Unnamed: 52')
            if i == 0:
                final_dataset = salt.copy()
            else:
                final_dataset = pandas.concat([final_dataset, salt]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

            final_dataset.to_csv(f'data/preprocessed/{self.filename}_4000Line2_salt.csv')

        return final_dataset
    

dataset = Dataset4000Line2('data/raw/Режимный лист МУИС - 4000 Март 2024.xlsx')

parsing_pages = [f'{val:02}.03.2024' for val in range(1, 30)]

data_full = dataset.create_dataset_salt(parsing_pages)
data_salt = dataset.create_dataset(parsing_pages)