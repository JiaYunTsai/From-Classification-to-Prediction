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

    df["MA_5"] = df["收盤價(元)"].rolling(5).mean()
    df.loc[:, '相較前一天的波動'] = df['MA_5'].pct_change()
    
    sigma_pos_avg = df[df["相較前一天的波動"] > 0 ]["相較前一天的波動"].median()#.quantile(0.25)
    sigma_neg_avg = df[df["相較前一天的波動"] < 0 ]["相較前一天的波動"].median()#.quantile(0.75) #因為是負的，所以取0.75
    print("sigma_pos_avg: ", sigma_pos_avg)
    print("sigma_neg_avg: ", sigma_neg_avg)

    df.loc[:, '漲跌'] = df['相較前一天的波動'].apply(
        lambda x: '上漲' if x > sigma_pos_avg else '下跌' if x < sigma_neg_avg else '無')    
    
    return df
    

def main():
    df_22_23_filter = filter_target_stock_data()
    df_22_23_filter = clean_target_stock_data(df_22_23_filter)
    print(df_22_23_filter.loc[:, ['年月日', '收盤價(元)', 'MA_5', '相較前一天的波動']].head(10))
    print(df_22_23_filter.head(10))
    print(df_22_23_filter['漲跌'].value_counts())
    df_22_23_filter.to_csv('bda2023_mid_dataset/stock_data_2019_2023_filter_clean.csv', encoding='utf_8_sig' , index=False)
    

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