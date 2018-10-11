import numpy as np
from pylab import *


def plot_confusion_matrix(conf_arr, alphabet, path):
    cm_normalized = conf_arr.astype('float') / conf_arr.sum(axis=1)[:, np.newaxis]
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(12, 12), dpi=150)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Reds,
                    interpolation='nearest')
    width = len(conf_arr)
    height = len(conf_arr[0])


    ind_array = np.arange(len(alphabet))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01 :
            if c < 0.5:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=7, va='center', ha='center')


    # cb = fig.colorbar(res)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    locs, labels = plt.xticks(range(width), alphabet[:width])
    for t in labels:
         t.set_rotation(90)

    plt.yticks(range(height), alphabet[:height])

    plt.xlabel('Predicted Class Rank', fontsize=15)
    plt.ylabel('Actual Class Rank', fontsize=15)

    plt.savefig(path + 'confusion_matrix.eps', format='eps')
    plt.savefig(path + 'confusion_matrix.png')
