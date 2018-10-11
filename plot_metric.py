import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PATH = 'results/'
clfs = ['logisticregression/', 'naive_bayes/', 'svm/']
grams = ['1gram/', '2gram/', '3gram/']


accuracy = []
f1 = []
precision = []
recall = []
auc = []

for clf in clfs:
    for gram in grams:
        path = PATH + clf + gram + 'metrics'
        f = open(path)
        lines = f.readlines()
        accuracy.append(float(lines[0].strip().split()[1]))
        f1.append(float(lines[1].strip().split()[1]))
        precision.append(float(lines[2].strip().split()[1]))
        recall.append(float(lines[3].strip().split()[1]))
        auc.append(float(lines[4].strip().split()[1]))




x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
y = f1
temp = zip(x, y)
# 在柱状图上显示具体数值, ha水平对齐, va垂直对齐
items = ['LR_1', 'LR_2', 'LR_3',
         'NB_1', 'NB_2', 'NB_3',
         'SVM_1', 'SVM_2', 'SVM_3']

plt.bar(x, y)# 折线 1 x 2 y 3 color
for x, y in zip(x, y):
    plt.text(x + 0.05, y + 0.0005, '%.4f' % y, ha = 'center', va = 'bottom')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], items, rotation=0)
plt.ylim(0.95, 1.01)

plt.xlabel(u'classifiers with n_gram',fontproperties='SimHei',fontsize=14)
plt.ylabel(u'f1',fontproperties='SimHei',fontsize=14)
plt.savefig('results/' + 'f1.png', format='png')
plt.show()


# def convert(x):
#     if type(x) != str:
#         print(x)
#     l = x.split('?')
#     if len(l) < 2:
#         return ""
#     else:
#         pre, path = x.split('?')[0], x.split('?')[1]
#         key_value_list = path.split('&')
#         keys = []
#         for tupe in key_value_list:
#             keys.append(tupe.split('=')[0])
#         return ''.join(pre.split('/')[3:] + keys)
#
#
# df = pd.read_csv('dataset/uri2.csv')
# df.fillna(method='pad', inplace=True)
# df['uri'] = df['uri'].apply(convert)
#
# df.to_csv('dataset/new_uri2.csv', index=False)
