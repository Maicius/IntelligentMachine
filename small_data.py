import pandas as pd
from main import remove_no_float

small_data = pd.read_excel('raw_data/small.xlsx')
print(small_data.shape)
small_data.drop(['ID'], axis=1, inplace=True)
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
wrong_row = wrong_data1[wrong_data1.na_count >= 3].row.values
print(small_data.shape)
small_data.drop(wrong_row, axis=0, inplace=True)
print(small_data.shape)
wrong_data2 = wrong_data1[wrong_data1 > lower]


