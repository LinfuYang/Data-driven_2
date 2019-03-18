from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

from coding.Bo_model import opt
from coding import gbdt_model

data_train = pd.read_csv('../dataset/training.csv')
data_test = pd.read_csv('../dataset/testing_validation.csv')
column_data = list(data_train.columns)

# 数据处理，类别数据

data_train['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_train['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

data_test['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_test['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

column = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
             'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',  'Tdewpoint', 'NSM',  'WeekStatus', 'Day_of_week']


for index in column:
    scaler = StandardScaler()
    data_train[index] = scaler.fit_transform((data_train[index].values).reshape(-1, 1))
for index in column:
    scaler = StandardScaler()
    data_test[index] = scaler.fit_transform((data_test[index].values).reshape(-1, 1))

column_LM = ['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
             'NSM',  'WeekStatus', 'Day_of_week']
x_train = data_train[column]
x_test = data_test[column]

y_train = data_train['Appliances']
y_test = data_test['Appliances']



best_x, best_y = opt(algo=gbdt_model, max_iters=200, train_x=x_train, train_y=y_train ,cv_mun=10, init_start=10,
                     model_score='neg_mean_squared_error', print_opt=True)


print(best_x)
print(best_y)