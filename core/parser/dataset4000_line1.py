import pandas
import datetime

from typing import List 

from base import Base


class Dataset4000Line1(Base):
    def __init__(self, filename):
          super().__init__(filename)


    def get_muis_4000_line_1(self, df: pandas.DataFrame) -> pandas.DataFrame:
        index_to_keep = 3
        # Строки, начиная с найденного индекса
        df_line_one = df.iloc[index_to_keep:index_to_keep+12]

        result = df_line_one.iloc[:, 0:49]
        result.dropna(subset=['        Режимный лист МУИС-4000 техн. линия № 1 Воронцовского месторождения'], inplace=True)

        return result


    def get_moisture_4000_line_1(self, df: pandas.DataFrame, column: str) -> pandas.DataFrame:
        mask = (df[column] == 'Показания влагомера')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]+2:index_to_keep[0]+15]
        result = df_first_rows.iloc[1: , :]

        return result


    def get_fresh_water_4000_line_1(self, df: pandas.DataFrame, column: str) -> pandas.DataFrame:
        mask = (df[column] == 'Расход пресной воды')
        index_to_keep = df[mask]
        index_to_keep = index_to_keep.index
        print(index_to_keep)
        print(len(index_to_keep))
        # Удалить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep[0]+3:index_to_keep[0]+15]
        result = df_first_rows

        return result


    def get_salt(self, df: pandas.DataFrame, column_time: str) -> pandas.DataFrame:
        index_to_keep = 2

        mask_last_idx = (df[column_time] == 'ТЛ-1 тн')
        last_index_to_keep = df[mask_last_idx]
        last_index_to_keep = last_index_to_keep.index
        print(last_index_to_keep)
        # Удалить строки, начиная с найденного индекса
        df_first_rows = df.iloc[index_to_keep:last_index_to_keep[0]]
        result = df_first_rows

        return result


    def rename_cols(self, df: pandas.DataFrame, df_type: str) -> pandas.DataFrame:
        if df_type == 'muis_4000_line_1':
            df.columns = [
                'Дата',  'Блок манифольда|P бм  |кгс/см²',
                'Блок манифольда|t жидкости| °С', 'С-1/1|P сеп. |кгс/см²',
                'С-1/1|t жидкости |°С', 'С-1/1|L жидкости| см', 'С-1/2|P сеп. |кгс/см²',
                'С-1/2|t жидкости |°С', 'С-1/2|L жидкости| см','С-2/1|P сеп. |кгс/см²',
                'С-2/1|t жидкости|°С', 'С-2/1|L межфазный|см', 'С-2/1|L нефти|см',
                'С-2/2|P сеп. |кгс/см²', 'С-2/2|t жидкости|°С', 'С-2/2|L межфазный|см',
                'С-2/2|L нефти|см', 'ОН-1/1|P отс. |кгс/см²', 'ОН-1/1|t жидкости|°С',
                'ОН-1/1|L межфазный|см', 'ОН-1/2|P отс. |кгс/см²',
                'ОН-1/2|t жидкости|°С', 'ОН-1/2|L  межфазный|см', 'С-3/1|t жидкости|°С',
                'С-3/1|L нефти|см', 'С-3/2|t жидкости|°С',
                'С-3/2|L нефти|см', 'П-1|Т нефти на входе|°С', 'П-1|Т нефти на выходе|°С',
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

        elif df_type == 'water_4000_line_1':
            df.columns = [
                'Дата',
                'Добыча воды за 2 ч',
                'Добыча воды за 2 ч, расч',
                'Добыча воды за 2 ч, расч, по ТОРам']

        elif df_type == 'moisture_4000_line_1':
            df.columns = [
                'Дата',
                'Показания влагомера',
            ]

        elif df_type == 'fresh_water_4000_line_1':
            df.columns = [
                'Расход пресной воды'
            ]

        elif df_type == 'salt_4000_line_1':
            df.columns = [
                'Дата', 'Место сбора', 'Сод. Cl соед., мг/дм3'
            ]

        return df

    def replace_with_time(self, value):
        if isinstance(value, str):
            value = value.replace(' ', '')
            return datetime.datetime.strptime(value, "%H:%M").time()
        return value


    def clear_dataset_4000_line_1_indicators(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_indicator_4000 = df.copy()
        df_indicator_4000 = df_indicator_4000.iloc[:, 0: 49]
        df_indicator_4000 = df_indicator_4000.drop(['Unnamed: 1'], axis=1)
        df_indicator_4000 = self.get_muis_4000_line_1(df_indicator_4000)
        df_indicator_4000 = self.rename_cols(df_indicator_4000, 'muis_4000_line_1')
        df_indicator_4000 = df_indicator_4000.reset_index()
        df_indicator_4000 = df_indicator_4000.drop(['index'], axis=1)
        df_indicator_4000['Дата'] = df_indicator_4000['Дата'].apply(self.replace_with_time)
        df_indicator_4000 = self.fill_date(df_indicator_4000)
        # df_indicator_4000 = self.drop_each_non_exist_hour(df_indicator_4000)
        df_indicator_4000 = df_indicator_4000.dropna()

        return df_indicator_4000


    def clear_dataset_4000_line_1_water_hourly(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_water_hourly = df.copy()
        df_water_hourly = df_water_hourly.iloc[:, [0, 61, 63, 64]]
        df_water_hourly = self.get_muis_4000_line_1(df_water_hourly)
        df_water_hourly = df_water_hourly.reset_index()
        df_water_hourly = df_water_hourly.drop(['index'], axis=1)
        df_water_hourly = self.rename_cols(df_water_hourly, 'water_4000_line_1')
        # df_water_hourly = self.drop_each_non_exist_hour(df_water_hourly)
        df_water_hourly = df_water_hourly.drop(['Дата'], axis=1)

        return df_water_hourly


    def clear_dataset_moisture_4000_line_1(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_moisture = df.copy()
        df_moisture = df_moisture.iloc[:, [0, 93]]
        try:
            df_moisture = self.get_moisture_4000_line_1(df_moisture, 'Unnamed: 93')
        except:
            df_moisture = self.get_moisture_4000_line_1(df_moisture, 'Вода с БКНС.1')
        df_moisture = df_moisture.dropna()
        df_moisture = self.rename_cols(df_moisture, 'moisture_4000_line_1')
        df_moisture = df_moisture.reset_index(drop=True)
        df_moisture = self.fill_date(df_moisture)
        # df_moisture = self.drop_each_non_exist_hour(df_moisture)
        df_moisture = df_moisture.drop(['Дата'], axis=1)
        df_moisture = df_moisture.dropna()

        return df_moisture


    def clear_dataset_4000_line_1_fresh_water(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df_fresh_water = df.copy()
        df_fresh_water = df_fresh_water.iloc[:, [55]]

        df_fresh_water = self.get_fresh_water_4000_line_1(df_fresh_water, 'Unnamed: 55')
        df_fresh_water = self.rename_cols(df_fresh_water, 'fresh_water_4000_line_1')
        df_fresh_water = df_fresh_water.reset_index()
        df_fresh_water = df_fresh_water.drop(['index'], axis=1)
        df_fresh_water = df_fresh_water.dropna()

        return df_fresh_water


    def clear_dataset_4000_line_1_salt(self, df: pandas.DataFrame, pages: List[str], i: int) -> pandas.DataFrame:
        df_salt = df.iloc[:, [75, 76, 78]]
        df_salt = self.get_salt(df_salt, 'Время')
        df_salt = df_salt.dropna()
        date: str = pages[i]

        df_salt['Дата'] = date
        df_salt['Дата'] = pandas.to_datetime(df_salt['Дата'], format='%d.%m.%Y')
        df_salt['Время'] = pandas.to_datetime(df_salt['Время'], format='%H:%M:%S').dt.time

        # Объединение колонки 'date' и 'time' в новую колонку 'datetime'
        df_salt['Время'] = df_salt['Дата'] + pandas.to_timedelta(df_salt['Время'].astype(str))
        df_salt = df_salt.drop(['Дата'], axis=1)
        df_salt = self.rename_cols(df_salt, 'salt_4000_line_1')

        return df_salt


    def create_dataset(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()
        for i, page in enumerate(parsing_pages):
            df_day = self.xl.parse(page)
            indicators = self.clear_dataset_4000_line_1_indicators(df_day)
            water_hourly = self.clear_dataset_4000_line_1_water_hourly(df_day)
            moisture = self.clear_dataset_moisture_4000_line_1(df_day)
            fresh_water = self.clear_dataset_4000_line_1_fresh_water(df_day)

            result = pandas.concat([indicators, water_hourly, moisture, fresh_water], axis=1)
            if i == 0:
                final_dataset = result.copy()
            else:
                final_dataset = pandas.concat([final_dataset, result]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

            final_dataset = self.check_dataset(final_dataset)
            final_dataset.to_csv(f'data/preprocessed/{self.filename}_clear_Line1.csv')

        return final_dataset


    def create_dataset_salt(self, parsing_pages: List[str]) -> pandas.DataFrame:
        final_dataset = pandas.DataFrame()
        for i, page in enumerate(parsing_pages):
            df_day = self.xl.parse(page)
            salt = self.clear_dataset_4000_line_1_salt(df_day, parsing_pages, i)
            if i == 0:
                final_dataset = salt.copy()
            else:
                final_dataset = pandas.concat([final_dataset, salt]).copy()
                final_dataset = final_dataset.reset_index(drop=True)

            final_dataset.to_csv(f'data/preprocessed/{self.filename}_4000Line1_salt.csv')

        return final_dataset
    

dataset = Dataset4000Line1('data/raw/Режимный_лист_МУИС_4000_Апрель_2024.xlsx')

parsing_pages = [f'{val:02}.04.2024' for val in range(1, 23)]

data_full = dataset.create_dataset_salt(parsing_pages)
data_salt = dataset.create_dataset(parsing_pages)