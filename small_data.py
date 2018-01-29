import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from main import remove_waste_col, remove_miss_row, remove_miss_col, knn_fill_nan, ensemble_model_feature
from main import remove_no_float
import numpy as np
from sklearn import preprocessing
from sklearn import utils


def remove_wrong_row(small_data):
    nan_data1 = small_data.isnull()
    nan_data2 = nan_data1.sum(axis=0)
    nan_data3 = nan_data2.reset_index()
    nan_data3.columns = ['col', 'nan_count']
    nan_data = nan_data3.sort_values(by='nan_count')
    nan_data_value = nan_data[nan_data.nan_count > 5].col.values
    small_data.drop(nan_data_value, axis=1, inplace=True)
    data = remove_no_float(small_data)
    small_data = small_data[data]
    small_data.fillna(small_data.mean(), inplace=True)

    data1 = small_data.append(small_data.mean() + 3 * small_data.std(), ignore_index=True)
    data2 = data1.append(small_data.mean() - 3 * small_data.std(), ignore_index=True)
    upper = small_data.mean() + 3 * small_data.std()
    lower = small_data.mean() - 3 * small_data.std()

    wrong_data1 = (small_data > upper).sum(axis=1).reset_index()
    wrong_data1.columns = ['row', 'na_count']
    wrong_row1 = wrong_data1[wrong_data1.na_count >= 3].row.values

    wrong_data2 = (small_data < lower).sum(axis=1).reset_index()
    wrong_data2.columns = ['row', 'na_count']
    wrong_row2 = wrong_data2[wrong_data2.na_count >= 3].row.values
    wrong_row = np.concatenate((wrong_row1, wrong_row2))

    print(small_data.shape)
    small_data.drop(wrong_row, axis=0, inplace=True)
    print(wrong_row1)
    wrong_data2 = wrong_data1[wrong_data1 > lower]
    print(small_data.shape)


def change_object_to_float(data):
    print(data.shape)
    data_type = data.dtypes.reset_index()
    data_type.columns = ['col', 'dtype']
    set_object = set('A')
    dict_object = {}
    data_object_col = data_type[data_type.dtype == 'object'].col.values
    data_object = data[data_object_col]
    i = 0.0
    for object in data_object:
        set_object = set(data_object[object].values) | set_object
        print(set_object)
    for item in set_object:
        dict_object[item] = i
        i += 1.0
    print(dict_object)
    for col in data_object_col:
        for i in range(len(data[col].values)):
            data[col].values[i] = dict_object[data[col].values[i]]
    # for col in data_object_col:
    #     data[col].values = list(map(lambda x: dict_object[x], list(data[col].values)))

    # data_object.apply(lambda x: dict_object[x])

    # data.to_excel("half_data/data_to_float.xlsx")
    return data


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


def stack_data():
    y = pd.read_excel('raw_data/small.xlsx')
    y = y['Y'].values
    print(y.shape)
    y_a = pd.read_csv('raw_data/test_a_ans.csv', header=None)
    y_a = y_a[1].values
    print(y_a.shape)
    Y = np.vstack((y, y_a))
    print(Y)

if __name__ == '__main__':
    small_data = pd.read_excel('raw_data/small.xlsx')
    print(small_data.shape)

    small_data.drop(['ID'], axis=1, inplace=True)
    # remove_wrong_row(small_data)
    small_data = change_object_to_float(small_data)
    # # small_data.fillna(small_data.median(), inplace=True)
    small_data = remove_waste_col(small_data)
    # # x_train = small_data.drop(['Y'], axis=1)

    # # x_train = do_lda(x_train.values, y_train.values)
    small_data = remove_miss_col(small_data)
    small_data = remove_miss_row(small_data)
    small_data = knn_fill_nan(small_data, 9)
    small_data.fillna(small_data.median(), inplace=True)
    print(np.any(np.isnan(small_data)))
    print(np.any(np.isfinite(small_data)))
    Y = small_data['Y']
    small_data.drop(['Y'], axis=1, inplace=True)
    print(small_data.shape, Y.shape)
    # small_data.to_excel('small_data2.xlsx')
    # stack_data()
    ensemble_model_feature(small_data, Y, 10)
