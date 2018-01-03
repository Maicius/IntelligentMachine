import pandas as pd
from main import cal_MSE
yesterday = pd.read_csv('测试A_答案模板.csv', header=None)
today = pd.read_csv('final2.csv', header=None)
yesterday = yesterday[1]
today = today[1]
print(cal_MSE(yesterday, today))
pass

