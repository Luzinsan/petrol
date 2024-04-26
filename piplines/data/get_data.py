import os
import pandas as pd
import sys



if __name__ == "__main__":  
    year = 2022
    print(os.environ['RAW_DATA_PATH'])
    raw_path = os.environ['RAW_DATA_PATH']
    dataset = pd.read_csv(f'{raw_path}/{year}/Режимный лист МУИС 2022.xlsx1250_clear.csv', 
                          parse_dates=['Дата']).drop(columns='Unnamed: 0')

    dataset.columns = [
        "Date","BlockP","BlockT",
        "C1P","C1T","C1L",
        "C2P","C2T","C2interfacial_L","С2petrol_L",

        "OH_T","OH_interfacial_L","OH_P",
        
        "C3T","C3L",
        "P1Tinput","P1Toutput","P1Pinput","P1Poutput","P1Тcoolant","P1Tgases","P1Рburner",
        "P2Tinput","P2Toutput","P2Pinput","P2Poutput","P2Tcoolant","P2Tgases","P2Pburner",
        "BEV1L","BEV1V_water","BEV1V_petrol","BEV1T_water"
    ]
    vlagomer = pd.read_csv(f'{raw_path}/{year}/1250_vlagomer.csv', parse_dates=['Дата']).drop(columns='Unnamed: 0')
    vlagomer.columns = ['Date', 'Water', 'Vlagomer']

    dataset = dataset.merge(vlagomer)
    init_path =  os.environ['INITIAL_DATA_PATH']
    dataset = dataset.drop(columns=['Date', 'BEV1L', 'BEV1V_water', 'BEV1V_petrol', 'BEV1T_water'])

    dataset.to_csv(f'{init_path}/{year}/1250.csv', index=False)
    
    