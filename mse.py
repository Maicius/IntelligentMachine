import pandas as pd
import numpy as np
from main import cal_MSE
yesterday = pd.read_csv('result/result.csv', header=None)
today = pd.read_csv('final3.csv', header=None)
yesterday = yesterday[1]
today = today[1]
print(cal_MSE(yesterday, today))
pass

corr_num = np.round(np.linspace(0.1, 0.2, 20), 2)
print(corr_num)
train_data = pd.read_csv('x_train.csv')
print(train_data.shape)


