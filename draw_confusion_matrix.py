from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from confusion_matrix import plot_confusion_matrix
from time import time
import pandas as pd
import numpy as np
import os

alphabet = ['9GAME', 'ALIPAY', 'AMAP', 'BAICIZHAN', 'BAIDU', 'DIANPING', 'DOUYU',
 'EASTMONEY', 'HIIDO', 'HUYA', 'IFENG', 'IMMOMO', 'IQIYI', 'JD', 'KG.QQ',
 'KUGOU', 'KUWO', 'MGTV', 'NEWS.QQ', 'PINDUODUO', 'QINGTING', 'QQMUSIC',
 'QUNAR', 'SUNING', 'TAOBAO', 'TOUTIAO', 'VIDEO.QQ', 'WEIBO', 'WOSTORE',
 'XIAMI', 'XIMALAYA', 'XUNLEI', '360ASSISTANT', 'LAVF', 'QQBROWSER', 'QYPLAYER']



root_dir = 'result/'

grams = ['1_gram/', '2_gram/', '3_gram/']

for gram in grams:
    path = root_dir + gram + 'GradientBoostingClassifier/'
    df = pd.read_csv(path + 'predict_data.csv')
    conf_arr = confusion_matrix(df['y_ture'].values, df['y_pred'].values)
    kw = {
        'conf_arr':conf_arr,
        'alphabet':alphabet,
        'path':path,
    }
    plot_confusion_matrix(**kw)