import pandas as pd
import numpy as np
from main import cal_MSE
yesterday = pd.read_csv('result/测试A_答案模板.csv', header=None)
today = pd.read_csv('result/xgboost-0.42.csv', header=None)
yesterday = yesterday[1]
today = today[1]
print(cal_MSE(yesterday, today))
pass

# corr_num = np.round(np.linspace(0.1, 0.2, 20), 2)
# print(corr_num)
# train_data = pd.read_csv('x_train.csv')
# print(train_data.shape)

offsets = np.arange(394, 400, 2)
print(sorted(offsets, reverse=True))
print(offsets[0])


