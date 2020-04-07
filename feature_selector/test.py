from __future__ import division
import pandas as pd
import os
import numpy as np
from feature_selector import FeatureSelector

str_list = []
for i in range(0, 16):
        str_list.append(str(i))

x_train = pd.DataFrame(columns=str_list)

train_path = '/home/xiaofeng.zhang/Workplace/test_cn_nn/all_data/'

PATH = os.getcwd()

y_train = []

train_data = os.listdir(train_path)
count = 0

for sample in train_data:
    count = count + 1
    print(count)
    file_path = train_path + sample

    f_s = sample.split('__')

    frame_id = int(f_s[2][6:])

    dist_to_ego = f_s[3]
    ego_relation = os.path.splitext(f_s[4])[0]

    if ego_relation == "cut_out" or ego_relation == "ego":
        continue

    x_r = []

    with open(file_path) as f:
        x = np.genfromtxt(file_path,
                          dtype=float,
                          invalid_raise=True,
                          missing_values="NaN",
                          skip_footer=1,
                          filling_values="0")
        fr = f.readlines()

        label = -1
        if (fr[10].strip('\n') == 'ST'):
            label = 0
        if (fr[10].strip('\n') == 'LC'):
            label = 1
        if (fr[10].strip('\n') == 'RC'):
            label = 1

        if len(x) == 10 and label != -1:
            y_train.append(label)
            for i in range(0, 10):
                x_r.append(x[0][i] - x[9][i])

    a_series = pd.Series(x_r, index=x_train.columns)
    x_train = x_train.append(a_series, ignore_index=True)

fs = FeatureSelector(data = x_train, labels = y_train)

fs.identify_missing(missing_threshold=0.6)
missing_features = fs.ops['missing']
print('miss_features')
for f in missing_features:
    print(f)

fs.missing_stats.head(10)

fs.identify_single_unique()
single_unique = fs.ops['single_unique']

print('single_unique')
for f in single_unique:
    print(f)

fs.identify_collinear(correlation_threshold=0.975)
correlated_features = fs.ops['collinear']

print('correlated_features')
for f in correlated_features:
    print(f)

fs.identify_zero_importance(task = 'classification', eval_metric = 'auc',
                            n_iterations = 10, early_stopping = True)

zero_importance_features = fs.ops['zero_importance']
print('zero_importance_features')
for f in zero_importance_features:
    print(f)

fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']
print('low_importance_features')
for f in low_importance_features:
    print(f)