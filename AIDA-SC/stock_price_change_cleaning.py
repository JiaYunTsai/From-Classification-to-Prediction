#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

def filter_target_stock_data():
    try:
        df_22_23_filter = pd.read_csv('bda2023_mid_dataset/stock_data_2019_2023_filter.csv')
    except:
        df = pd.read_excel('bda2023_mid_dataset/stock_data_2019-2023.xlsx', sheet_name = ['上市2023', '上市2022'])
        df_2023 = df['上市2023'][df['上市2023']['證券代碼'] == "2308 台達電"]
        df_2022 = df['上市2022'][df['上市2022']['證券代碼'] == "2308 台達電"]

        df_22_23_filter = pd.concat([df_2023, df_2022], axis=0)
        df_22_23_filter.to_csv('bda2023_mid_dataset/stock_data_2019_2023_filter.csv', encoding='utf_8_sig' , index=False)
    return df_22_23_filter


def clean_target_stock_data(df):
    # 將年月日轉換成 datetime 型態
    df['年月日'] = pd.to_datetime(df['年月日'])
    df['星期'] = df['年月日'].dt.dayofweek
    df['星期'] = df['星期'].replace([4, 3, 2, 1, 0],['星期五', '星期四', '星期三', '星期二', '星期一'])
    df = df.sort_values(by='年月日', ascending=True)

    friday_df = df.loc[df['年月日'].dt.weekday==4].copy()
    friday_df.loc[:, '相較前一天的波動'] = friday_df['收盤價(元)'].pct_change()
    friday_df.loc[:, '漲跌'] = friday_df['相較前一天的波動'].apply(
        lambda x: '上漲' if x > 0.025 else '下跌' if x < -0.05 else '無')

    df = pd.merge(df, friday_df[['年月日', '漲跌']], on='年月日', how='left')
    df['漲跌'] = df['漲跌'].fillna(method='bfill')

    return df
    

def main():
    df_22_23_filter = filter_target_stock_data()
    df_22_23_filter = clean_target_stock_data(df_22_23_filter)
    print(df_22_23_filter.head(20))
    print(df_22_23_filter['漲跌'].value_counts())

if __name__ == "__main__":
    main()



# ['證券代碼', 
#  '年月日', 
#  '開盤價(元)', 
#  '最高價(元)', 
#  '最低價(元)', 
#  '收盤價(元)', 
#  '成交量(千股)', 
#  '成交值(千元)', 
#  '成交筆數(筆)', 
#  '流通在外股數(千股)', 
#  '本益比-TSE', 
#  '股價淨值比-TSE']