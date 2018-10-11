import pandas as pd
import numpy as np

labels = ['9game', 'alipay', 'amap', 'baicizhan', 'baidu', 'dianping', 'douyu',
 'eastmoney', 'hiido', 'huya', 'ifeng', 'immomo', 'iqiyi', 'jd', 'kg.qq',
 'kugou', 'kuwo', 'mgtv', 'news.qq', 'pinduoduo', 'qingting', 'qqmusic',
 'qunar', 'suning', 'taobao', 'toutiao', 'video.qq', 'weibo', 'wostore',
 'xiami', 'ximalaya', 'xunlei', 'zhushou.360', 'lavf', 'qqbrowser', 'qyplayer']

def get_dataset(n_gram):

    train_dir = 'train_vector_cross/'
    test_dir = 'test_vector_cross/'
    ngram_dir = ['', '1_gram', '2_gram', '3_gram']
    ngram = ngram_dir[n_gram]
    x_tr, y_tr, x_test, y_test = [],[],[],[]
    for label, name in enumerate(labels):

        train = pd.read_csv(train_dir + ngram + '/' + name +'.csv').values
        test = pd.read_csv(test_dir + ngram + '/' + name +'.csv').values

        x_tr.append(train)
        x_test.append(test)

        if label == 0:
            y_tr.append(np.zeros([len(train)], dtype=np.int))
            y_test.append(np.zeros([len(test)], dtype=np.int))
        else:
            y_tr.append(np.ones([len(train)], dtype=np.int) * label)
            y_test.append(np.ones([len(test)], dtype=np.int) * label)

    return np.concatenate(x_tr), np.concatenate(y_tr), np.concatenate(x_test), np.concatenate(y_test)


