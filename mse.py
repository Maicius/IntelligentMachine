import pandas as pd
import numpy as np
from main import cal_MSE
import matplotlib.pyplot as plt

# yesterday = pd.read_csv('result/submitB_A3-0.0265-0.03778.csv', header=None)
# today = pd.read_csv('result/submitB_A5.csv', header=None)
# yesterday = yesterday[1]
# today = today[1]
# print(cal_MSE(yesterday, today))

# corr_num = np.round(np.linspace(0.01, 0.5, 10), 2)
# print("corrnum", corr_num)
# train_data = pd.read_csv('half_data/x_train.csv')
# print(train_data.shape)
#
offsets = np.logspace(-2, 3, 200)
print(sorted(offsets, reverse=True))
print(offsets[0])

x_test = pd.read_csv('half_data/x_test.csv')
x_test_null = x_test.isnull().sum(axis=0).reset_index()
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
fourth = pd.read_csv('result/xgboost-0.0264-0.03771.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
fifth = pd.read_csv('result/xgboost4-0.02437-0.04045.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
sixth = pd.read_csv('result/submitB_A6-0.02604.csv', header=None).reset_index().drop([0], axis=1, inplace=False)

ensemble_result = 0.2 * first + 0.2 * second + 0.1 * third + 0.2 * fourth + 0.2 * fifth + 0.1 * sixth
# ensemble_result.to_csv('result/ensemble_submit.csv', header=None)
seventh = pd.read_csv('result/xgboost4-0.02456.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
eighth = pd.read_csv('result/submitB_A6-0.022799.csv', header=None).reset_index().drop([0], axis=1, inplace=False)

ensemble_result2 = 0.4 * ensemble_result + 0.3 * seventh + 0.3 * eighth

sub_df = pd.read_csv('raw_data/answer_A.csv', header=None)
sub_df['Value'] = ensemble_result2[1]
sub_df.to_csv('result/ensemble.csv', header=None, index=False)

plt.plot(first['index'], first[1], 'r')
# plt.plot(second['index'], second[1], 'black')
# plt.plot(third['index'], third[1], 'm')
# plt.plot(fifth['index'], fifth[1], 'b')
# plt.plot(sixth['index'], sixth[1], 'k')
# plt.plot(fourth['index'], fourth[1], 'g')
plt.plot(ensemble_result2['index'], ensemble_result2[1], 'b')
plt.plot(ensemble_result['index'], ensemble_result[1], 'black')
print(cal_MSE(second[1], fifth[1]))
plt.legend(loc='upper left')
plt.show()

def ensemble_submit():
    ridge1 = pd.read_csv('result/submitB_B6-0.02264.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    ridge2 = pd.read_csv('result/submitB_B6-0.02287.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    xgboost1 = pd.read_csv('result/xgboost4-0.02485.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    xgboost2 = pd.read_csv('result/xgboost4-0.0247.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    ensemble_result = 0.2 * ridge1 + ridge2 * 0.3 + xgboost1 * 0.2 + xgboost2 * 0.3
    sub_df = pd.read_csv('raw_data/answer_sample_b_20180117.csv', header=None)
    sub_df['Value'] = ensemble_result[1]
    sub_df.to_csv('result/ensembleB.csv', header=None, index=False)

def ensemble_last():
    ridge1 = pd.read_csv('result/submitB_B8.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    ridge2 = pd.read_csv('result/submitB_B7-0.0224.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    xgboost1 = pd.read_csv('result/submitB_B7-0.0248.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    xgboost2 = pd.read_csv('result/xgboost4-0.02576.csv', header=None).reset_index().drop([0], axis=1, inplace=False)
    ensemble_result = 0.2 * ridge1 + ridge2 * 0.3 + xgboost1 * 0.2 + xgboost2 * 0.3
    sub_df = pd.read_csv('raw_data/answer_sample_b_20180117.csv', header=None)
    sub_df['Value'] = ensemble_result[1]
    sub_df.to_csv('result/ensembleB2.csv', header=None, index=False)

if __name__ == '__main__':
    ensemble_last()