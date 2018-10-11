from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os

labels = ['9game', 'alipay', 'amap', 'baicizhan', 'baidu', 'dianping', 'douyu',
 'eastmoney', 'hiido', 'huya', 'ifeng', 'immomo', 'iqiyi', 'jd', 'kg.qq',
 'kugou', 'kuwo', 'mgtv', 'news.qq', 'pinduoduo', 'qingting', 'qqmusic',
 'qunar', 'suning', 'taobao', 'toutiao', 'video.qq', 'weibo', 'wostore',
 'xiami', 'ximalaya', 'xunlei', 'zhushou.360', 'lavf', 'qqbrowser', 'qyplayer']

train_data_dir = 'train_vector_cross/'
test_data_dir = 'test_vector_cross/'

ngram_dir = ['1_gram', '2_gram', '3_gram']

data_csv = './data.csv'

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def gen_ngram_vector(ngram):

    check_path(train_data_dir)
    check_path(test_data_dir)


    ngram_range = {
        3:(3,3),
        2:(2,2),
        1:(1,1)}[ngram]
    df_tr = pd.read_csv('dataset/train_dup.csv', encoding='utf8')
    df_test = pd.read_csv('dataset/test_dup.csv', encoding='utf8')

    df_tr.fillna('pad', inplace=True)
    df_test.fillna('pad', inplace=True)

    param = {
        'ngram_range' : ngram_range,
        'decode_error' : 'ignore',
        'token_pattern' : r'\b\w+\b',
        'analyzer' : 'char',
    }
    vectorizer = CountVectorizer(**param)
    x = pd.concat([df_tr, df_test], axis=0)
    vectorizer.fit(x['x'].values)

    for label in df_tr['label'].unique():
        tr_index = df_tr[df_tr['label'] == label].index
        tr = df_tr.iloc[tr_index]['x'].values

        test_index = df_test[df_test['label'] == label].index
        test = df_test.iloc[test_index]['x'].values

        tr = vectorizer.transform(tr).toarray()
        test = vectorizer.transform(test).toarray()
        tr[tr != 0] = 1
        test[test != 0] = 1

        tr_df = pd.DataFrame(tr)
        test_df = pd.DataFrame(test)



        check_path(train_data_dir + ngram_dir[ngram-1])
        check_path(test_data_dir + ngram_dir[ngram-1])
        tr_path = train_data_dir + ngram_dir[ngram-1] + '/' + labels[label] + '.csv'
        test_path = test_data_dir + ngram_dir[ngram-1] + '/' + labels[label] + '.csv'

        tr_df.to_csv(tr_path, index=False)
        test_df.to_csv(test_path, index=False)


if __name__ == '__main__':
    gen_ngram_vector(3)