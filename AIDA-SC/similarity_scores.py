import pandas as pd
import numpy as np
import csv
import monpa
from monpa import utils
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

def up_down_data_split(stock_df, df, keywords):
    stock_df['年月日'] = pd.to_datetime(stock_df['年月日']).dt.date

    df = df.rename(columns={'post_time': '年月日'})
    df['年月日'] = pd.to_datetime(df['年月日']).dt.date

    df = df[df['content'].str.contains(keywords, na=False)]
    df = pd.merge(df, stock_df[['年月日', '漲跌']], on='年月日', how='left')
    df = df.reset_index(drop=True)

    df_up = df.loc[df['漲跌'] == '上漲']
    df_down = df.loc[df['漲跌'] == '下跌']

    print('上漲', df_up.shape)
    print('下跌', df_down.shape)

    df_up.to_csv('bda2023_mid_dataset/df_up.csv', encoding='utf_8_sig' , index=False)
    df_down.to_csv('bda2023_mid_dataset/df_down.csv', encoding='utf_8_sig' , index=False)

    return df_up, df_down
    
def monpa_split(query):
    sentence_list = utils.short_sentence(query)
    cut_text = ' '.join(term.strip() for item in sentence_list for term in monpa.cut(item) 
                        if len(term.strip()) > 1)
    return cut_text

def get_content_list(df):
    data = np.array(df)

    content_list = []
    for row in data:
        query = f"{row[5]} {row[7]}"
        cut_text = monpa_split(query)
        content_list.append(cut_text)

    return content_list

def get_tf_counter_df(df, up_down):

    cut_text_list = get_content_list(df)
    
    cut_text_all = ' '.join(cut_text_list)
    cut_text_all_list = cut_text_all.split(" ")

    tf_counter=Counter(cut_text_all_list)

    tf_counter = pd.DataFrame(tf_counter.most_common(100), columns=['term', 'count'])
    tf_counter.to_csv(f'bda2023_mid_dataset/limit_{up_down}_term.csv', encoding='utf_8_sig' , index=False)
    return tf_counter

def get_vocab_id_dict(tf_counter):
    vocab_id_dict = {} # tf_counter 前面要把沒鑑別力的字扣除
    #把df的 term 欄位的內容轉成list
    term_list = tf_counter['term'].tolist()
    for i, word in enumerate(term_list):
        vocab_id_dict[word] = i
    return vocab_id_dict


def get_idf_limit_model(df_up, df_down, content_list, query):
    df = pd.concat([df_up, df_down], axis=0, ignore_index=True)
    
    vocab_id_dict = get_vocab_id_dict(df)
    print(vocab_id_dict)
    
    vectorizer = TfidfVectorizer(vocabulary= vocab_id_dict ,use_idf=True)
    X = vectorizer.fit_transform(content_list)
    # print(X)

    sparse.save_npz("tmp/limit_model.npz", X)

    cut_text = monpa_split(query)
    q =vectorizer.fit_transform([cut_text])
    print(q[0:1])
    cosine_similarities = linear_kernel(q[0:1], X).flatten() #與給定文件集的向量做相似度計算
    print(cosine_similarities) #印出與每一篇的相似度做觀察

def main():
    stock_df = pd.read_csv('bda2023_mid_dataset/stock_data_2019_2023_filter_clean.csv')
    news_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_news_2022-2023.csv')
    forum_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_forum_2022-2023.csv')
    bbs_df = pd.read_csv('bda2023_mid_dataset/bda2023_mid_bbs_2022-2023.csv')

    df_merged = pd.concat([news_df, forum_df, bbs_df], axis=0, ignore_index=True)
    keywords = "台達電|2308"
    # keywords = "台達電|2308|電子零組件|汽車電子|電源管理|散熱|電源供應器"
    # keywords = "台達電|2308|電子零組件|汽車電子|電源管理|散熱|電源供應器|乾坤科技|晶睿通訊|力林科技光寶科|群電|康舒"

    df_up, df_down = up_down_data_split(stock_df,df_merged, keywords)
    print("data split done")

    df = pd.concat([df_up.head(10), df_down.head(10)], axis=0, ignore_index=True)
    content_list = get_content_list(df)    

    tf_counter_df_up = get_tf_counter_df(df_up.head(10),"up")
    tf_counter_df_down = get_tf_counter_df(df_down.head(10), "down")
    print("tf_counder done")
 
    query='台股加權指數在最近9個交易日，從最高到最低點，跌了2,544點，創下史上最快速的失速列車紀錄；\
    12日台股盤中急挫1,418點，市場衰鴻遍野，據統計，盤中最多曾有711檔個股觸及跌停、占上市櫃的四成比重，\
    最後仍有251檔收跌停，其中，陽明等15檔股價亮燈跌停，仍有7千張以上賣單高掛，貨櫃三雄均入榜。'

    get_idf_limit_model(tf_counter_df_up, tf_counter_df_down, content_list, query)
    print("idf_limit done")
    
if __name__ == "__main__":
    main()