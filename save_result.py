from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from confusion_matrix import plot_confusion_matrix
from time import time
import pandas as pd
import numpy as np

alphabet = ['9GAME', 'ALIPAY', 'AMAP', 'BAICIZHAN', 'BAIDU', 'DIANPING', 'DOUYU',
 'EASTMONEY', 'HIIDO', 'HUYA', 'IFENG', 'IMMOMO', 'IQIYI', 'JD', 'KG.QQ',
 'KUGOU', 'KUWO', 'MGTV', 'NEWS.QQ', 'PINDUODUO', 'QINGTING', 'QQMUSIC',
 'QUNAR', 'SUNING', 'TAOBAO', 'TOUTIAO', 'VIDEO.QQ', 'WEIBO', 'WOSTORE',
 'XIAMI', 'XIMALAYA', 'XUNLEI', '360ASSISTANT', 'LAVF', 'QQBROWSER', 'QYPLAYER']


def fit_and_save_result(x_train, y_train, x_test, y_ture, clf, path):
    t1 = time()
    clf.fit(x_train, y_train)
    t2 = time()
    test_one_hot = label_binarize(y_ture, np.arange(36))
    y_score = clf.predict_proba(x_test)
    t3 = time()
    y_pre = clf.predict(x_test)
    t4 = time()
    conf_arr = confusion_matrix(y_true=y_ture, y_pred=y_pre)
    accuracy = metrics.accuracy_score(y_true=y_ture, y_pred=y_pre)
    f1 = metrics.f1_score(y_true=y_ture, y_pred=y_pre, average='macro')
    precision = metrics.precision_score(y_true=y_ture, y_pred=y_pre, average='macro')
    recall = metrics.recall_score(y_true=y_ture, y_pred=y_pre, average='macro')
    auc = metrics.roc_auc_score(y_true=test_one_hot, y_score=y_score, average='macro')


    auc_all, f1_all, recall_all, precision_all, acc_all = [], [], [], [], []
    for label in range(36):
        def convert(x):
            if x == label:
                return 1
            else:
                return 0

        label = int(label)
        ture = y_ture
        pre = y_pre
        score = y_score[:, label]
        ture = [convert(x) for x in ture]
        pre = [convert(x) for x in pre]

        auc_each = metrics.roc_auc_score(ture, score)
        f1_each = metrics.f1_score(y_true=ture, y_pred=pre, average='binary')
        recall_each = metrics.recall_score(ture, pre)
        precision_each = metrics.precision_score(ture, pre)
        accuracy_each = metrics.accuracy_score(ture, pre)

        auc_all.append(auc_each)
        f1_all.append(f1_each)
        recall_all.append(recall_each)
        precision_all.append(precision_each)
        acc_all.append(accuracy_each)

    each_class_pd = pd.DataFrame({
        'auc':auc_all,
        'f1':f1_all,
        'recall':recall_all,
        'precision_all':precision_all,
        'acc':acc_all
    })

    each_class_pd.to_csv(path + 'each_class_metrics.csv', index=False)


    df_score = pd.DataFrame(y_score)
    df_score.to_csv(path + 'predict_proba.csv', index=False)

    df_result = pd.DataFrame({'y_ture' : y_ture, 'y_pred' : y_pre})
    df_result.to_csv(path + 'predict_data.csv', index=False)



    with open(path + 'metrics', 'w') as w:
        w.write('train_time: %s' % str(t2 - t1) + '\n')
        w.write('predict_time: %s' % str(t4 - t3) + '\n')
        w.write('accuracy: %s' % str(accuracy) + '\n')
        w.write('f1: %s' % str(f1) + '\n')
        w.write('precision: %s' % str(precision) + '\n')
        w.write('recall: %s' % str(recall) + '\n')
        w.write('auc: %s' % str(auc) + '\n')

    # 保存最优超参数
    with open(path + 'best_params', 'w') as w:
        w.write('\n'.join(['%s %s' % (key, str(value)) for key, value in clf.best_params_.items()]))

    # 画出混淆矩阵并保存
    plot_confusion_matrix(conf_arr, alphabet, path)