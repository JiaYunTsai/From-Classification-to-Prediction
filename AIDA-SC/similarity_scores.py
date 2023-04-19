# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv
import monpa
from monpa import utils
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
import modling


def up_down_data_split(stock_df, df, keywords):
    stock_df['年月日'] = pd.to_datetime(stock_df['年月日']).dt.date

    df = df.rename(columns={'post_time': '年月日'})
    df['年月日'] = pd.to_datetime(df['年月日']).dt.date

    df = df[df['content'].str.contains(keywords, na=False)]
    df = pd.merge(df, stock_df[['年月日', '漲跌']], on='年月日', how='left')
    df = df.reset_index(drop=True)

    df_up = df.loc[df['漲跌'] == 'upward_trend']
    df_down = df.loc[df['漲跌'] == 'downward_trend']

    print('Number of upward trend articles', df_up.shape)
    print('Number of downward trend articles', df_down.shape)

    df_up.to_csv('bda2023_mid_dataset/df_up.csv',
                 encoding='utf_8_sig', index=False)
    df_down.to_csv('bda2023_mid_dataset/df_down.csv',
                   encoding='utf_8_sig', index=False)

    return df_up, df_down


def monpa_split(query):
    sentence_list = utils.short_sentence(query)
    cut_text = ' '.join(term.strip() for item in sentence_list for term in monpa.cut(item)
                        if len(term.strip()) > 1)
    return cut_text


def get_content_set(df_up, df_down):
    content_set = {'upward_trend': [], 'downward_trend': []}

    for df in [df_up, df_down]:
        content_list = []
        for index, row in df.iterrows():
            query = f"{row['title']} {row['content']}"
            cut_text = monpa_split(query)
            content_list.append(cut_text)
        if df.equals(df_up):
            content_set['upward_trend'] = content_list
        else:
            content_set['downward_trend'] = content_list

    return content_set


def get_tf_counter_df(content_set, up_down):

    up_list = ' '.join(content_set['upward_trend']).split(" ")
    down_list = ' '.join(content_set['downward_trend']).split(" ")

    common_word_list = list(set(up_list).intersection(set(down_list)))

    up_list = [word for word in up_list if word not in common_word_list]
    down_list = [word for word in down_list if word not in common_word_list]

    up_tf_counter = Counter(up_list).most_common(2000)
    down_tf_counter = Counter(down_list).most_common(2000)

    print(
        f"Top 10 up word list{ up_tf_counter[:10]} from {len(up_list)} up_list\n")
    print(
        f"Top 10 down word list{ down_tf_counter[:10]} from {len(down_list)} up_list\n")

    tf_counter = down_tf_counter + up_tf_counter

    tf_counter = pd.DataFrame(tf_counter, columns=['term', 'count'])
    print(tf_counter.shape)
    tf_counter.to_csv(
        f'bda2023_mid_dataset/limit_{up_down}_term.csv', encoding='utf_8_sig', index=False)
    return tf_counter


def get_vocab_id_dict(tf_counter):
    vocab_id_dict = {}
    term_list = tf_counter['term'].tolist()
    for i, word in enumerate(term_list):
        vocab_id_dict[word] = i

    return vocab_id_dict


def get_idf_limit_model(df, content_set, query):
    vocab_id_dict = get_vocab_id_dict(df)
    # vocab_id_dict = {'this': 0, 'is': 1, 'a': 2, 'sample': 3, 'sentence': 4}
    # content_list = [    'This is a sample sentence.',    'This is another example sentence.',    'Here is a third sentence for testing.']

    vectorizer = TfidfVectorizer(vocabulary=vocab_id_dict, use_idf=True)
    content_list = content_set['upward_trend'] + content_set['downward_trend']
    X = vectorizer.fit_transform(content_list)

    # print(pd.DataFrame(X.toarray(),columns=vocab_id_dict))

    sparse.save_npz("tmp/limit_model.npz", X)
    cut_text = monpa_split(query)
    q = vectorizer.fit_transform([cut_text])

    cosine_similarities = linear_kernel(q[0:1], X).flatten()  # 與給定文件集的向量做相似度計算
    related_docs_indices = cosine_similarities.argsort()  # 將相似度由小至大做排序，並轉換成文件編號
    print(related_docs_indices)

    return X, q


def get_y_label(content_set):
    upward_count = len(content_set["upward_trend"])
    down_count = len(content_set["downward_trend"])
    y = []
    y.extend(['看漲'] * upward_count)
    y.extend(['看跌'] * down_count)
    return y


def main():
    stock_df = pd.read_csv(
        'bda2023_mid_dataset/stock_data_2019_2023_filter_clean.csv')
    news_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_news_2022-2023.csv')
    forum_df = pd.read_csv(
        'bda2023_mid_dataset/bda2023_mid_forum_2022-2023.csv')
    bbs_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_bbs_2022-2023.csv')

    df_merged = pd.concat([news_df, forum_df, bbs_df],
                          axis=0, ignore_index=True)
    # keywords = "台達電|2308"
    # keywords = "台達電|2308|電子零組件|汽車電子|電源管理|散熱|電源供應器"
    keywords = "台達電|2308|電子零組件|汽車電子|電源管理|散熱|電源供應器|乾坤科技|晶睿通訊|力林科技光寶科|群電|康舒"

    df_up, df_down = up_down_data_split(stock_df, df_merged, keywords)
    print("data split done!\n")

    content_set = get_content_set(df_up, df_down)
    # content_set = get_content_set(df_up.head(20), df_down.head(20))

    tf_counter_df = get_tf_counter_df(content_set, "none")
    print("tf_counder done!!!\n")

    query = '台股加權指數在最近9個交易日，從最高到最低點，跌了2,544點，創下史上最快速的失速列車紀錄；\
    12日台股盤中急挫1,418點，市場衰鴻遍野，據統計，盤中最多曾有711檔個股觸及跌停、占上市櫃的四成比重，\
    最後仍有251檔收跌停，其中，陽明等15檔股價亮燈跌停，仍有7千張以上賣單高掛，貨櫃三雄均入榜。'

    X, q = get_idf_limit_model(tf_counter_df, content_set, query)
    print("idf_limit done")

    y = get_y_label(content_set)

    print("train data check")
    result = modling.NB_predict(X, y, q)
    print("NB_predict", result)

    print("X", type(X), X.shape)
    print("y", len(y))
    print("\n==================================")
    print("start NB modling")
    modling.NB_modle(X, y)

    print("\n==================================")
    print("start DecisionTree modling")
    modling.DecisionTree_modle(X, y)

    print("\n==================================")
    print("start SVC modling")
    modling.SVC_modle(X, y)

    print("\n==================================")
    print("XG modling")
    y = [1 if i == "看漲" else 0 for i in y]
    modling.XGboost(X, y)

    print("\n==================================")
    print('RF_MODEL')
    modling.RF_model(X, y)

    print("\n==================================")
    print("GBC_MODEL")
    modling.GBC_model(X, y)


if __name__ == "__main__":
    main()
