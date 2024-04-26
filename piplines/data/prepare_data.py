from core.data import Dataset
import pandas as pd
import os



def prepare(dataset: pd.DataFrame) -> pd.DataFrame:
    df = Dataset(dataset)
    df.scale(df.cols[:-1], method='standard')
    df.recovery_outliers(df.cols[:-1], save_interval=[-6,10], insert=False)
    df.scale(df.cols[:-1], method='minmax')
    return df.df


if __name__ == "__main__":
    year = os.environ['YEAR']
    init_path = os.environ['INITIAL_DATA_PATH']
    df = pd.read_csv(f"{init_path}/{year}/1250.csv")
    df = df.astype('float32')
    prepared_df = prepare(df)
    pred_path = os.environ['PREPARED_DATA_PATH']
    prepared_df.to_csv(f'{pred_path}/{year}/1250.csv', index=False)
