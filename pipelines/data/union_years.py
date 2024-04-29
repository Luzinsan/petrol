from core.data import Dataset
import pandas as pd
import os
import sys
from pathlib import Path


if __name__ == "__main__":
    year_1 = sys.argv[1]
    year_2 = sys.argv[2]
    prep_path = os.environ['PREPARED_DATA_PATH']
    df_1 = pd.read_csv(f"{prep_path}/{year_1}/1250.csv")
    df_2 = pd.read_csv(f"{prep_path}/{year_2}/1250.csv")

    df_1_2 = pd.concat([df_1, df_2])

    path_save = f'{prep_path}/{year_1}{year_2}'
    Path(path_save).mkdir(parents=True, exist_ok=True)
    df_1_2.to_csv(f'{path_save}/1250_not_recovered.csv', index=False)
