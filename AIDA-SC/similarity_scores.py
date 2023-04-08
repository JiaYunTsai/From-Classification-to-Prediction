import pandas as pd
import numpy as np

def news_data_cleaning(df):
    df = df.rename(columns={'post_time': '年月日'})
    df['年月日'] = pd.to_datetime(df['年月日']).dt.date

    return df
    

def main():
    stock_df = pd.read_csv('bda2023_mid_dataset/stock_data_2019_2023_filter_clean.csv')
    news_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_news_2022-2023.csv')
    forum_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_forum_2022-2023.csv')
    bbs_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_bbs_2022-2023.csv')

    stock_df = news_data_cleaning(stock_df)
    bbs_df = news_data_cleaning(bbs_df)
    forum_df = news_data_cleaning(stock_df)

    keywords = "台達電|電子零組件|電源供應器|乾坤科技|晶睿通訊|力林科技|汽車電子|電源管理|散熱|光寶科|群電|康舒"

    bbs_df['content'] = bbs_df['content'].str.contains(keywords)
    bbs_df = bbs_df[bbs_df['content']]
    
    df = pd.merge(bbs_df, stock_df[['年月日', '漲跌']], on='年月日', how='left')
    df = df.reset_index(drop=True)

    bbs_up_df = df.loc[df['漲跌'] == '上漲']
    bbs_down_df = df.loc[df['漲跌'] == '下跌']

    bbs_up_df.to_csv('bda2023_mid_dataset/bbs_up_df.csv', encoding='utf_8_sig' , index=False)
    bbs_down_df.to_csv('bda2023_mid_dataset/bbs_down_df.csv', encoding='utf_8_sig' , index=False)
    print(bbs_up_df.shape)
    print(bbs_down_df.shape)

    
if __name__ == "__main__":
    main()