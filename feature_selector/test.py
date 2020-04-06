import pandas as pd
from __future__ import division
import os
import numpy as np
from feature_selector import FeatureSelector

str_list = []
for i in range(0, 16):
    for j in range (0, 10):
        str_list.append(str(i)+'_'+str(j))

x_train = pd.DataFrame(columns=str_list)

train_path = '/home/xiaofeng.zhang/Workplace/train_cn/all_data/'

PATH = os.getcwd()

y_train = []

train_data = os.listdir(train_path)

for sample in train_data:
    file_path = train_path + sample

    f_s = sample.split('__')

    frame_id = int(f_s[2][6:])
    if frame_id <= 100:
        continue

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
            for sublist in x:
                for item in sublist:
                    x_r.append(item)

    a_series = pd.Series(x_r, index=x_train.columns)
    x_train = x_train.append(a_series, ignore_index=True)

fs = FeatureSelector(data = x_train, labels = y_train)
fs.identify_all(selection_params = {'missing_threshold': 0.5, 'correlation_threshold': 0.7,
                                    'task': 'regression', 'eval_metric': 'l2',
                                     'cumulative_importance': 0.9})