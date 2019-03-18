from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
data_train = pd.read_csv('../dataset/training.csv')
from sklearn.decomposition import PCA
data_test = pd.read_csv('../dataset/testing_validation.csv')


column_data = list(data_train.columns)
# ['date', 'Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
# 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2', 'NSM', 'WeekStatus', 'Day_of_week']


# 数据处理，类别数据

data_train['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_train['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

data_test['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_test['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

column = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
             'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',  'Tdewpoint', 'NSM',  'WeekStatus', 'Day_of_week']


# for index in column:
#     scaler = StandardScaler()
#     data_train[index] = scaler.fit_transform((data_train[index].values).reshape(-1, 1))
# for index in column:
#     scaler = StandardScaler()
#     data_test[index] = scaler.fit_transform((data_test[index].values).reshape(-1, 1))

column_LM = ['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
             'NSM',  'WeekStatus', 'Day_of_week']
x_train = data_train[column]
x_test = data_test[column]

y_train = data_train['Appliances']
y_test = data_test['Appliances']
m, n = np.shape(x_train)
pca = PCA(n_components=5)
pca.fit(x_train)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
'''
x_train_new = pca.fit_transform(x_train)
x_test_new = pca.fit(x_test)

'''

'''
# 线性回归做交叉验证
# cv_lr_model = LinearRegression()
# lr_best_score = np.mean(np.sqrt(-(cross_val_score(cv_lr_model, x_train, y=y_train, scoring='neg_mean_squared_error', cv=10))))
# print(lr_best_score)



model_lr = LinearRegression()
model_lr.fit(x_train, y_train)
pro = model_lr.predict(x_test)

R2_lr = r2_score(y_test, pro)

RMSE_lr = np.sqrt(mean_squared_error(y_test, pro))

MAE_lr = median_absolute_error(y_test, pro)

print('****************'+'LinearRegression'+'****************')
print('RMSE_lr:', RMSE_lr)
print('R2_lr:', R2_lr)
print('MAE_lr:', MAE_lr)





model_svr = SVR()
model_svr.fit(x_train, y_train)
pro = model_svr.predict(x_test)

R2_svr = r2_score(y_test, pro)

RMSE_svr = np.sqrt(mean_squared_error(y_test, pro))

MAE_svr = median_absolute_error(y_test, pro)

print('****************'+'SVR'+'****************')
print('RMSE_svr:', RMSE_svr)
print('R2_svr:', R2_svr)
print('MAE_svr:', MAE_svr)


model_gbdt = GradientBoostingRegressor()
model_gbdt.fit(x_train, y_train)
pro_gbdt = model_gbdt.predict(x_test)


R2_gbdt = r2_score(y_test, pro_gbdt)

RMSE_gbdt = np.sqrt(mean_squared_error(y_test, pro_gbdt))

MAE_gbdt = median_absolute_error(y_test, pro_gbdt)

print('****************'+'GradientBoostingRegressor'+'****************')
print('RMSE_gbdt:', RMSE_gbdt)
print('R2_gbdt:', R2_gbdt)
print('MAE_gbdt:', MAE_gbdt)


model_rf = RandomForestRegressor()
model_rf.fit(x_train, y_train)
pro_rf = model_rf.predict(x_test)

R2_rf = r2_score(y_test, pro_rf)

RMSE_rf = np.sqrt(mean_squared_error(y_test, pro_rf))

MAE_rf = median_absolute_error(y_test, pro_rf)

print('****************'+'RandomForestRegressor'+'****************')
print('RMSE_rf:', RMSE_rf)
print('R2_rf:', R2_rf)
print('MAE_rf:', MAE_rf)


'''
