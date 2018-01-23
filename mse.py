import pandas as pd
import numpy as np
from main import cal_MSE
import matplotlib.pyplot as plt

# yesterday = pd.read_csv('result/submitB_A3-0.0265-0.03778.csv', header=None)
# today = pd.read_csv('result/submitB_A5.csv', header=None)
# yesterday = yesterday[1]
# today = today[1]
# print(cal_MSE(yesterday, today))
# pass
#


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


def plot_image(x, y, x_label=None, y_label=None):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_image2(x, y, x_label=None, y_label=None):
    plt.plot(x, y)
    plt.title("cross val score")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

first = pd.read_csv('result/submitB_A2-0.03620.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
second = pd.read_csv('result/submitB_A3-0.0245-0.03778.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
third = pd.read_csv('result/submitB_A5-0.022075-0.04593.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
fourth = pd.read_csv('result/xgboost-0.0264.csv', header=None).reset_index().drop([0], axis=1, inplace=False)


plt.plot(first['index'], first[1], 'r')
plt.plot(second['index'], second[1], 'black')
plt.plot(third['index'], third[1], 'g')
plt.plot(fourth['index'], fourth[1], 'b')
plt.show()