import pandas as pd
import numpy as np
from main import cal_MSE
yesterday = pd.read_csv('result/submitB_A3-0.0265-0.03778.csv', header=None)
today = pd.read_csv('result/submitB_A5.csv', header=None)
yesterday = yesterday[1]
today = today[1]
print(cal_MSE(yesterday, today))
pass

# corr_num = np.round(np.linspace(0.01, 0.5, 10), 2)
# print("corrnum", corr_num)
# train_data = pd.read_csv('half_data/x_train.csv')
# print(train_data.shape)
#
offsets = np.logspace(-2, 3, 200)
print(sorted(offsets, reverse=True))
print(offsets[0])

x_test = pd.read_csv('half_data/x_test.csv')
print(np.any(np.isnan(x_test)))
print(np.any(np.isfinite(x_test)))
