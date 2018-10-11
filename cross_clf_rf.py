from get_data import get_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from confusion_matrix import plot_confusion_matrix
from time import time
from save_result import fit_and_save_result
import numpy as np
import os


n_gram = 3
estimator = RandomForestClassifier
# 参数范围
kw = {
    'n_estimators' : [290],
    'max_depth' : [90],
    'max_leaf_nodes' : [70],

}


alphabet = ['9game', 'alipay', 'amap', 'baicizhan', 'baidu', 'dianping', 'douyu',
 'eastmoney', 'hiido', 'huya', 'ifeng', 'immomo', 'iqiyi', 'jd', 'kg.qq',
 'kugou', 'kuwo', 'mgtv', 'news.qq', 'pinduoduo', 'qingting', 'qqmusic',
 'qunar', 'suning', 'taobao', 'toutiao', 'video.qq', 'weibo', 'wostore',
 'xiami', 'ximalaya', 'xunlei', 'zhushou.360', 'lavf', 'qqbrowser', 'qyplayer']

ngrams = ['', '1_gram', '2_gram', '3_gram']


path = 'cross_result/' + ngrams[n_gram] + '/' + estimator.__name__ + '/'

if not os.path.exists(path): os.makedirs(path)

x_train, y_train, x_test, y_test = get_dataset(n_gram)

clf = GridSearchCV(
        estimator(),
        kw,
        scoring='f1_macro',
        cv=5,
        verbose=1,
        n_jobs=5)


fit_and_save_result(x_train, y_train, x_test, y_test, clf, path)
