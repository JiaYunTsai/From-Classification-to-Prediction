import pandas as pd
import numpy as np
import csv
import monpa
from monpa import utils
from collections import Counter

def up_down_data_split(stock_df, df, keywords):
    stock_df['年月日'] = pd.to_datetime(stock_df['年月日']).dt.date

    df = df.rename(columns={'post_time': '年月日'})
    df['年月日'] = pd.to_datetime(df['年月日']).dt.date

    df = df[df['content'].str.contains(keywords, na=False)]
    df = pd.merge(df, stock_df[['年月日', '漲跌']], on='年月日', how='left')
    df = df.reset_index(drop=True)

    df_up = df.loc[df['漲跌'] == '上漲']
    df_down = df.loc[df['漲跌'] == '下跌']

    print(df_up.shape)
    print(df_down.shape)

    df_up.to_csv('bda2023_mid_dataset/df_up.csv', encoding='utf_8_sig' , index=False)
    df_down.to_csv('bda2023_mid_dataset/df_down.csv', encoding='utf_8_sig' , index=False)

    return df_up, df_down
    
def monpa_split(df, up_down):
    tf_counter=Counter()
    df_counter=Counter()
    data = np.array(df)

    for row in data:
        query = f"{row[5]} {row[7]}"
        sentence_list = utils.short_sentence(query)
        df_tmp=Counter()
    
        terms = (term.strip() for item in sentence_list for term in monpa.cut(item) if len(term.strip()) > 1)
        tf_counter.update(terms)

        df_tmp.update({term: 1 for term in set(terms) if df_tmp[term] == 0})
        df_counter+=df_tmp     
    # for tern, count in tf_counter.most_common(100):
    #     print(tern, count)

    # for tern, count in df_counter.most_common(100):
    #     print(tern, count)

    tf_counter = pd.DataFrame(tf_counter.most_common(100), columns=['term', 'count'])
    tf_counter.to_csv(f'bda2023_mid_dataset/limit_{up_down}_tern.csv', encoding='utf_8_sig' , index=False)
    return tf_counter


def main():
    stock_df = pd.read_csv('bda2023_mid_dataset/stock_data_2019_2023_filter_clean.csv')
    news_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_news_2022-2023.csv')
    forum_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_forum_2022-2023.csv')
    bbs_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_bbs_2022-2023.csv')

    df_merged = pd.concat([news_df, forum_df, bbs_df], axis=0, ignore_index=True)
    keywords = "台達電|電子零組件|電源供應器|乾坤科技|晶睿通訊|力林科技|汽車電子|電源管理|散熱|光寶科|群電|康舒"

    df_up, df_down = up_down_data_split(stock_df,df_merged, keywords)

    df_up = monpa_split(df_up.head(10),"up")
    df_down = monpa_split(df_down.head(10), "down")
    # df_up 與 df_down term 欄位 相同的詞直接刪除
    df_up = df_up[~df_up['term'].isin(df_down['term'])]

    print(df_up)
    print(df_down)

    
if __name__ == "__main__":
    main()