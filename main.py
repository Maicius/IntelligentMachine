# coding=utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import cross_validation
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn import utils
from sklearn import ensemble


def pre_process_data():
    print("begin to read data")
    train_data = pd.read_excel('raw_data/训练_20180117.xlsx')
    x_test = pd.read_excel('raw_data/测试A_20180117.xlsx')

    # remove ID column
    train_data.drop(['ID'], axis=1, inplace=True)
    x_test.drop(['ID'], axis=1, inplace=True)
    print("raw_data:", train_data.shape, x_test.shape)
    # 去掉缺失值过多的行
    train_data = remove_miss_row(train_data)
    print("remove miss row:", train_data.shape)

    # 去掉缺失值过多的列
    train_data = remove_miss_col(train_data)
    print("remove miss col:", train_data.shape)

    # 填补缺失值
    train_data.fillna(train_data.median(axis=0), axis=0, inplace=True)
    print("填充缺失值:" + str(train_data.shape))

    # 将字母属性转化为浮点数
    train_data = change_object_to_float(train_data)

    # 去掉日期列以及数据过大的列
    train_data = remove_waste_col(train_data)
    print("去掉日期列以及数据过大的列:", train_data.shape)

    # 去除不符合正太分布的行
    x_train = remove_wrong_row(train_data)
    print("去除不符合正太分布的行数:", x_train.shape)

    # 分割标签与数据
    y_train = x_train['Value']
    x_train.drop(['Value'], axis=1, inplace=True)
    print("分割数据", y_train.shape, x_train.shape)

    # 计算皮尔森系数，降低维度
    corr_df = cal_corrcoef(x_train, y_train)
    corr02 = corr_df[corr_df.corr_value >= 0.21]
    corr02_col = corr02['col'].values.tolist()
    # 获取最终训练集与测试集
    x_train = train_data[corr02_col]
    x_test = x_test[corr02_col]
    print("去除皮尔森系数较小的维数:", y_train.shape, x_train.shape)

    # 填充测试集中的缺失值
    x_test.fillna(x_test.median(axis=0), axis=0, inplace=True)
    x_test = change_object_to_float(x_test)

    print("Finish preprocess")
    print(x_train.shape, y_train.shape)
    return x_train, y_train, x_test


# 去除缺失值过多的列
def remove_miss_col(data):
    nan_data = data.isnull().sum(axis=0).reset_index()
    nan_data.columns = ['col', 'nan_count']
    nan_data = nan_data.sort_values(by='nan_count')
    nan_data_value = nan_data[nan_data.nan_count > 40].col.values
    data.drop(nan_data_value, axis=1, inplace=True)
    return data


#  删除非数字列
def remove_no_float(data):
    data_type = data.dtypes.reset_index()
    data_type.columns = ['col', 'dtype']
    data_object = data_type[data_type.dtype == 'object'].col.values
    data_object = data[data_object]
    data_object.to_csv('half_data/non_float_col.csv', index=False)
    return data_type[data_type.dtype == 'float64'].col.values


def fill_null(data):
    pass


def change_object_to_float(data):
    data_type = data.dtypes.reset_index()
    data_type.columns = ['col', 'dtype']
    set_object = set('A')
    dict_object = {}
    data_object_col = data_type[data_type.dtype == 'object'].col.values
    data_object = data[data_object_col]
    i = 0.0
    for object in data_object:
        set_object = set(data_object[object].values) | set_object
    for item in set_object:
        dict_object[item] = i
        i += 1.0
    # print(dict_object)
    for col in data_object_col:
        for i in range(len(data[col].values)):
            data[col].values[i] = dict_object[data[col].values[i]]
    return data


# 计算协方差
def cal_corrcoef(float_df, y_train):
    corr_values = []
    float_col = list(float_df.columns)
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values.astype(float), y_train) \
                                   [0, 1]))
    corr_df = pd.DataFrame({'col': float_col, 'corr_value': corr_values})
    corr_df = corr_df.sort_values(by='corr_value', ascending=False)
    return corr_df


# 去掉数字相同的列以及日期列
def remove_waste_col(data):
    columns = list(data.columns)
    date_col = []
    for col in columns:
        max_num = data[col].max()
        if max_num != data[col].min() and max_num < 1e13 and str(max_num).find('2017') == -1 and str(max_num).find(
                '2016') == -1:
            date_col.append(col)
    return data[date_col]


# 去除不符合正太分布的行
def remove_wrong_row(data):
    upper = data.mean(axis=0) + 3 * data.std(axis=0)
    lower = data.mean(axis=0) - 3 * data.std(axis=0)

    wrong_data1 = (data > upper).sum(axis=1).reset_index()
    wrong_data1.columns = ['row', 'na_count']
    # 参数是经过调试的
    wrong_row1 = wrong_data1[wrong_data1.na_count >= 40].row.values
    wrong_data2 = (data < lower).sum(axis=1).reset_index()
    wrong_data2.columns = ['row', 'na_count']
    wrong_row2 = wrong_data2[wrong_data2.na_count >= 95].row.values
    wrong_row = np.concatenate((wrong_row1, wrong_row2))

    data.drop(wrong_row, axis=0, inplace=True)
    return data


def remove_miss_row(data):
    miss_row = data.isnull().sum(axis=1).reset_index()
    miss_row.columns = ['row', 'miss_count']
    # 移除缺失值大于500的行
    miss_row_value = miss_row[miss_row.miss_count >= 500].row.values
    data.drop(miss_row_value, axis=0, inplace=True)
    return data


def normalize_data(data):
    return data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # return preprocessing.scale(data, axis=0)
    # return data.apply(lambda x: (x - np.average(x)) / np.std(x))


def create_model(x_train, y_train, alpha):
    print("begin to train...")
    model = Ridge(alpha=alpha)
    clf1 = ensemble.BaggingRegressor(model, n_jobs=1, n_estimators=900)
    clf2 = ensemble.AdaBoostRegressor(n_estimators=900, learning_rate=0.01)
    clf3 = ensemble.RandomForestRegressor(n_estimators=900)
    clf4 = ensemble.ExtraTreesRegressor(n_estimators=900)
    # print("Bagging")

    scores = -cross_validation.cross_val_score(model, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    scores1 = -cross_validation.cross_val_score(clf1, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    scores2 = -cross_validation.cross_val_score(clf2, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    scores3 = -cross_validation.cross_val_score(clf3, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    scores4 = -cross_validation.cross_val_score(clf4, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    #
    print('=========================')
    print('Scores:')
    print(scores.mean())
    print(scores1.mean())
    print(scores2.mean())
    print(scores3.mean())
    print(scores4.mean())
    clf1.fit(x_train, y_train)
    print("Finish")
    return clf1


def cal_MSE(y_predict, y_real):
    n = len(y_predict)
    print("样本数量:" + str(n))
    return np.sum(np.square(y_predict - y_real)) / n


def find_min_alpha(x_train, y_train):
    alphas = np.logspace(-2, 3, 200)
    # print(alphas)
    test_scores = []
    alpha_score = []
    for alpha in alphas:
        clf = Ridge(alpha)
        test_score = -cross_validation.cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
        test_scores.append(np.mean(test_score))
        alpha_score.append([alpha, np.mean(test_score)])
    print("final test score:")
    print(test_scores)
    print(alpha_score)

    sorted_alpha = sorted(alpha_score, key=lambda x: x[1], reverse=False)
    print(sorted_alpha)
    alpha = sorted_alpha[0][0]
    print("best alpha:" + str(alpha))
    return alpha


def search_cv(x_train, y_train, x_test):
    xgb_model = xgb.XGBModel()
    params = {
        'booster': ['gblinear'],
        'silent': [1],
        'learning_rate': [x for x in np.round(np.linspace(0.01, 0.5, 10), 2)],
        'reg_lambda': [lambd for lambd in np.logspace(0, 3, 50)],
        'objective': ['reg:linear']
    }
    print('begin')
    clf = GridSearchCV(xgb_model, params,
                       scoring='neg_mean_squared_error',
                       refit=True)

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    sub_df = pd.read_csv('raw_data/answer_A.csv', header=None)
    sub_df['Value'] = preds
    sub_df.to_csv('result/xgboost3.csv', header=None, index=False)
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw RMSE:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


def find_best_offset():
    pass


def plot_image(x, y, x_label=None, y_label=None):
    plt.plot(x, y)
    plt.title("cross val score")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def do_lda(x_train, y_train):
    print("Begin LDA。。。")
    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(y_train)
    print(utils.multiclass.type_of_target(y_train))
    print(utils.multiclass.type_of_target(encoded))
    print(encoded)
    lda = LinearDiscriminantAnalysis(n_components=10)
    lda.fit(x_train, encoded)
    x_train_new = lda.transform(x_train)
    # x_train_new.to_csv('lda_train.csv', header=None, index=False)
    print(x_train_new)
    return x_train_new


def train_with_LR_L2(x_train, y_train, x_test, alpha):
    model = create_model(x_train, y_train, alpha)
    print("交叉验证...")
    scores = cross_validation.cross_val_score(model, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    print(scores)
    print("mean:" + str(scores.mean()))
    ans = model.predict(x_test)
    sub_df = pd.read_csv('raw_data/answer_A.csv', header=None)
    sub_df['Value'] = ans
    sub_df.to_csv('result/submitB_A5.csv', header=None, index=False)


if __name__ == '__main__':
    # 数据预处理，特征工程
    x_train, y_train, x_test = pre_process_data()
    # 保存特征工程的结果到文件
    x_train.to_csv('half_data/x_train.csv', header=None, index=False)
    y_train.to_csv('half_data/y_train.csv', header=None, index=False)
    x_test.to_csv('half_data/x_test.csv', header=None, index=False)
    # 从文件中读取经过预处理的数据
    x_train = pd.read_csv('half_data/x_train.csv', header=None)
    y_train = pd.read_csv('half_data/y_train.csv', header=None)
    print(x_train.shape, y_train.shape)
    x_test = pd.read_csv('half_data/x_test.csv', header=None)
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    X = np.vstack((x_train, x_test))
    # normalize数据
    X = preprocessing.scale(X)
    x_train = X[0:len(x_train)]
    x_test = X[len(x_train):]
    print("开始训练:")
    print(x_train.shape, y_train.shape)
    # 寻找L2正则的最优化alpha
    alpha = find_min_alpha(x_train, y_train)
    # 训练模型
    train_with_LR_L2(x_train, y_train, x_test, alpha)
    # xgboost 调参
    # search_cv(x_train, y_train, x_test)
