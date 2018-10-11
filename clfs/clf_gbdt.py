from get_data import get_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from confusion_matrix import plot_confusion_matrix
import numpy as np
import os


n_gram = 1
estimator = GradientBoostingClassifier

alphabet = ['9GAME', 'ALIPAY', 'AMAP', 'BAICIZHAN', 'BAIDU', 'DIANPING', 'DOUYU',
 'EASTMONEY', 'HIIDO', 'HUYA', 'IFENG', 'IMMOMO', 'IQIYI', 'JD', 'KG.QQ',
 'KUGOU', 'KUWO', 'MGTV', 'news.qq', 'pinduoduo', 'qingting', 'qqmusic',
 'qunar', 'suning', 'taobao', 'toutiao', 'video.qq', 'weibo', 'wostore',
 'xiami', 'ximalaya', 'xunlei', 'zhushou.360', 'lavf', 'qqbrowser', 'qyplayer']

ngrams = ['', '1_gram', '2_gram', '3_gram']


path = 'result/' + ngrams[n_gram] + '/' + estimator.__name__ + '/'

if not os.path.exists(path): os.makedirs(path)

x_train, y_train, x_test, y_test = get_dataset(n_gram)
# 参数范围
kw = {
    'max_depth':[5,],
    'n_estimators':[100],
}
clf = GridSearchCV(
        estimator(),
        kw,
        scoring='f1_macro',
        cv=5,
        verbose=1,
        n_jobs=5)
from save_result import fit_and_save_result
fit_and_save_result(x_train, y_train, x_test, y_test, clf, path)