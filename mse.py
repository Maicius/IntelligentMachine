import pandas as pd
import numpy as np
from main import cal_MSE
yesterday = pd.read_csv('result/submit_B2.csv', header=None)
today = pd.read_csv('result/submit_B3.csv', header=None)
yesterday = yesterday[1]
today = today[1]
print(cal_MSE(yesterday, today))
pass

corr_num = np.round(np.linspace(0.01, 0.5, 10), 2)
print("corrnum", corr_num)
train_data = pd.read_csv('half_data/x_train.csv')
print(train_data.shape)

offsets = np.arange(394, 400, 2)
print(sorted(offsets, reverse=True))
print(offsets[0])


