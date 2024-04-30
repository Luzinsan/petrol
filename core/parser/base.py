import datetime
import pandas


class Base:
    def __init__(self, filename: str) -> None:
      self.xl = pandas.ExcelFile(filename)
      self.filename = filename.split('/')[-1].split('.')[0]


    def drop_each_non_exist_hour(self, df: pandas.DataFrame) -> pandas.DataFrame:
      # Создаем список индексов для удаления
      indexes_to_drop = list(range(12, len(df)+1, 13))

      # Удаляем записи с заданными индексами
      df = df.drop(indexes_to_drop)
      return df


    def rename_cols(self, df: pandas.DataFrame, df_type: str) -> pandas.DataFrame:
      pass


    def is_datetime(self, x) -> bool:
      return isinstance(x, datetime.datetime)


    def replace_with_time(self, value):
          if isinstance(value, str):
            try:
              value = value.replace(' ', '')
              return datetime.datetime.strptime(value, "%H:%M").time()
            except:
              try:
                value = value.replace(';', ':')
                return datetime.datetime.strptime(value, "%H:%M").time()
              except:
                try:
                  value = value.replace('::', ':')
                  return datetime.datetime.strptime(value, "%H:%M").time()
                except:
                  try:
                    value = value.replace('.', ':')
                    return datetime.datetime.strptime(value, "%H:%M").time()
                  except:
                    value = '0' + value
                    return datetime.datetime.strptime(value, "%H.%M").time()
          return value


    def fill_date(self, df: pandas.DataFrame) -> pandas.DataFrame:
      datetime_records = df[df['Дата'].apply(self.is_datetime)].index
      for idx in datetime_records:
        date = df.loc[idx, 'Дата'].date()
        for i in range(idx+1, idx+14):
          try:
            time = df.loc[i, 'Дата']
            df.loc[i, 'Дата'] = datetime.datetime.combine(date, time)
          except: pass

      return df


    def check_dataset(self, df: pandas.DataFrame) -> pandas.DataFrame:
      duplicates_in_column = df[df.duplicated(subset=['Дата'], keep=False)].index
      if len(duplicates_in_column) > 0:
        print(f'Alert: дубликаты в дате - {duplicates_in_column}')

        df = df.drop_duplicates(subset=['Дата'], keep='last')

      if len(df.index) > 4700:
        print('Alert: подозрительно много записей')

      return df