from sklearn.model_selection import train_test_split

import pandas as pd
import os

file_path = 'input/'


def create_data():
    class_dict = dict()
    ret = []
    for name in os.listdir(file_path):
        label = len(class_dict)
        class_dict[name] = label
        for line in open(file_path + name):
            line = line.strip()
            if line.__len__() < 1:
                continue
            ret.append({'uri':line, 'label':label})
    df = pd.DataFrame(ret)
    df.to_csv('./dataset/uri.csv', index=False)
    with open('./dataset/class_dict', 'w') as f:
        f.write(str(class_dict))

def train_val_test_split():
    df = pd.read_csv('input/data_dup.csv')
    labels = df['label'].unique()
    print(len(labels))
    tr, test = [], []
    for label in labels:
        index = df[df['label'] == label].index

        df_labeled = df.iloc[index]

        tr_labeled, test_labeled = train_test_split(df_labeled, test_size=0.8, shuffle=True)

        tr.append(tr_labeled)
        test.append(test_labeled)

    tr_df = pd.concat(tr, axis=0)
    test_df = pd.concat(test, axis=0)
    tr_df.to_csv('dataset/train_dup.csv', index=False)
    test_df.to_csv('dataset/test_dup.csv', index=False)


if __name__ == '__main__':
    train_val_test_split()
