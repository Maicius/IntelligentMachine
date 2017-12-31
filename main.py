# encoding = utf-8

import pandas as pd


def pre_process_data():
    print("begin to read data")
    train_data = pd.read_excel('small.xlsx')
    # 去掉空行
    print("raw_data:" + str(train_data.shape))
    nan_data_value = remove_nan_data(train_data)

    train_data.drop(nan_data_value, axis=1, inplace=True)
    print("remove nan data:" + str(train_data.shape))
    float_data = remove_no_float(train_data)
    # train_data = remove_date(train_data)
    train_data = train_data[float_data]
    print(float_data)
    print(train_data.shape)


def remove_nan_data(data):
    nan_data = data.isnull().sum(axis=0).reset_index()
    nan_data.columns = ['col', 'nan_count']
    nan_data = nan_data.sort_values(by='nan_count')
    nan_data_value = nan_data[nan_data.nan_count > 20].col.values
    print(nan_data_value)
    return nan_data_value


#  删除非数字行
def remove_no_float(data):
    data_type = data.dtypes.reset_index()
    data_type.columns = ['col', 'dtype']
    return data_type[data_type.dtype == 'float64'].col.values


def remove_date(data, float_data):
    data_list = data.tolist()
    float_date = []
    for col in data_list:
        data_str = str(data[col].min())
        if data_str.startswith('2017') or data_str.startswith('2016'):
            float_date.append(col)
    date_df = data[float_date].col.values
    print("date shape:" + date_df.shape)
    print(date_df)
    return data.drop(date_df, axis=1, inplace=True)


if __name__ == '__main__':
    pre_process_data()
