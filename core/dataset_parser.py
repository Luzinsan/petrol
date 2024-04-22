import datetime
import pandas

class Dataset:
    def __init__(self, filename) -> None:
      self.xl = pandas.ExcelFile(filename)
      self.filename = filename

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
      # удаление дубликатов, т.к. иногда попадаются два одинаковых дня
      index_to_keep = index_to_keep.index
      # Удалить строки, начиная с найденного индекса
      df_first_rows = df.iloc[index_to_keep[0]:index_to_keep[0]+16]
      index_to_keep = index_to_keep.delete(0)
      df_new = pandas.concat([df.iloc[idx+1:idx+16] for idx in index_to_keep])
      frames = [df_first_rows, df_new]

      result = pandas.concat(frames)
      result = result.iloc[1: , :]
      result.dropna(subset=['Режимный лист МУИС-2500 Воронцовского месторождения'], inplace=True)

      return result

    def get_muis_1250(self, df):
      mask = (df['Режимный лист МУИС-2500 Воронцовского месторождения'] == 'Режимный лист МУИС-1250 Воронцовского месторождения') \
            # & (df['Unnamed: 2'] == 'Блок манифольда')
      index_to_keep = df[mask]
      # index_to_keep = index_to_keep[~index_to_keep.duplicated('Режимный лист МУИС-2500 Воронцовского месторождения', keep='last')]
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

    
    def get_moisture(self, df):
      mask = (df['Unnamed: 27'] == 'Показания влагомера')
      index_to_keep = df[mask]
      index_to_keep = index_to_keep.index
      print(index_to_keep)
      print(len(index_to_keep))
      # Удалить строки, начиная с найденного индекса
      df_first_rows = df.iloc[index_to_keep[0]+2:index_to_keep[0]+16]
      index_to_keep = index_to_keep.delete(0)
      df_new = pandas.concat([df.iloc[idx+3:idx+16] for idx in index_to_keep])
      frames = [df_first_rows, df_new]

      result = pandas.concat(frames)
      result = result.iloc[1: , :]

      return result


    def drop_each_non_exist_hour(self, df) -> pandas.DataFrame:
      # Создаем список индексов для удаления
      indexes_to_drop = list(range(12, len(df)+1, 13))

      # Удаляем записи с заданными индексами
      df = df.drop(indexes_to_drop)
      return df


    def rename_cols(self, df, df_type) -> pandas.DataFrame:
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

      elif df_type == 'muis_1250':
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

      return df


    def is_datetime(self, x):
      return isinstance(x, datetime.datetime)


    def fill_date(self, df) -> pandas.DataFrame:
      datetime_records = df[df['Дата'].apply(self.is_datetime)].index
      for idx in datetime_records:
        date = df.loc[idx, 'Дата'].date()
        for i in range(idx+1, idx+14):
          try:
            time = df.loc[i, 'Дата']
            df.loc[i, 'Дата'] = datetime.datetime.combine(date, time)
          except: pass

      return df


    def clear_dataset_moisture(self, df):
      df_moisture = df.copy()
      df_moisture = df_moisture.iloc[1855:, [0, 22, 27]]
      df_moisture = self.get_moisture(df_moisture)
      df_moisture = self.rename_cols(df_moisture, 'moisture')
      df_moisture = df_moisture.reset_index(drop=True)
      df_moisture = self.fill_date(df_moisture)
      df_moisture = self.drop_each_non_exist_hour(df_moisture)

      return df_moisture


    def clear_dataset_1250_indicators(self, df):
      df_indicator_1250 = df.copy()
      df_indicator_1250 = df_indicator_1250.iloc[:, 0: 35]
      df_indicator_1250 = df_indicator_1250.drop(['Unnamed: 1', 'Unnamed: 16'], axis=1)
      df_indicator_1250 = self.get_muis_1250(df_indicator_1250)
      df_indicator_1250 = self.rename_cols(df_indicator_1250, 'muis_1250')
      df_indicator_1250 = df_indicator_1250.reset_index()
      df_indicator_1250 = df_indicator_1250.drop(['index'], axis=1)
      df_indicator_1250 = self.fill_date(df_indicator_1250)
      df_indicator_1250 = self.drop_each_non_exist_hour(df_indicator_1250)
      df_indicator_1250 = df_indicator_1250.dropna()

      return df_indicator_1250

    def clear_dataset_2500_indicators(self, df, i) -> pandas.DataFrame:
      df_indicator = df.copy()
      df_indicator = df_indicator.iloc[:, 0: 45]
      df_indicator = df_indicator.drop(['Unnamed: 1', 'Unnamed: 23'], axis=1)
      df_indicator = self.get_muis_2500(df_indicator, i, True)
      df_indicator = self.rename_cols(df_indicator, 'muis_2500')
      df_indicator = df_indicator.reset_index()
      df_indicator = df_indicator.drop(['index'], axis=1)
      df_indicator = self.fill_date(df_indicator)
      df_indicator = self.drop_each_non_exist_hour(df_indicator)

      return df_indicator


    def clear_dataset_2500_water_hourly(self, df, i) -> pandas.DataFrame:
      df_water_hourly = df.copy()
      df_water_hourly = df_water_hourly.iloc[:, [0, 2, 57, 59, 60]]
      df_water_hourly = self.get_muis_2500(df_water_hourly, i)
      df_water_hourly = df_water_hourly.reset_index()
      df_water_hourly = df_water_hourly.drop(['index'], axis=1)
      df_water_hourly = self.rename_cols(df_water_hourly, 'water_2500')
      df_water_hourly = self.drop_each_non_exist_hour(df_water_hourly)
      df_water_hourly = df_water_hourly.drop(['Дата', 'Unnamed: 2'], axis=1)

      return df_water_hourly


    def check_dataset(self, df) -> pandas.DataFrame:
      duplicates_in_column = df[df.duplicated(subset=['Дата'], keep=False)].index
      if len(duplicates_in_column) > 0:
        print(f'Alert: дубликаты в дате - {duplicates_in_column}')

        df = df.drop_duplicates(subset=['Дата'], keep='last')

      if len(df.index) > 4700:
        print('Alert: подозрительно много записей')

      return df


    def create_dataset_2500(self, parsing_pages) -> pandas.DataFrame:

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
        final_dataset.to_csv(f'{self.filename}_clear.csv', index=False)

      return final_dataset

    def create_dataset_1250(self, parsing_pages) -> pandas.DataFrame:
      final_dataset = pandas.DataFrame()
      for i, page in enumerate(parsing_pages):
        df_month = self.xl.parse(page)
        indicators = self.clear_dataset_1250_indicators(df_month)
        if i == 0:
          final_dataset = indicators.copy()
        else:
          final_dataset = pandas.concat([final_dataset, indicators]).copy()
          final_dataset = final_dataset.reset_index(drop=True)

        final_dataset = self.check_dataset(final_dataset)
        final_dataset.to_csv(f'{self.filename}1250_clear.csv', index=False)

      return final_dataset


    def create_dataset_1250_moisture(self, parsing_pages) -> pandas.DataFrame:
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
        final_dataset.to_csv(f'{self.filename}1250_moisture_clear.csv', index=False)

      return final_dataset

dataset = Dataset('~/notebooks/petrol/data/raw/2023/Режимный_лист_МУИС_2023.xlsx')
parsing_pages = [
    'Январь 2023',
    'Февраль 2023',
    'Март 2023',
]
data_full_2500 = dataset.create_dataset_2500(parsing_pages)
data_full_1250 = dataset.create_dataset_1250(parsing_pages)
data_full_moisture = dataset.create_dataset_1250_moisture(parsing_pages)