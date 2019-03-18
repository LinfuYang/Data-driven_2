from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import lightgbm as lgb
import numpy as np
import random
from coding.ABO.AGPR import A_GPR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from GPy.models import GPRegression
from GPy.kern import RBF
import warnings
warnings.filterwarnings('ignore')


def pre_gp_mu_var(x_new, model, return_var=False):
    if return_var:
        mu, var = model.predict(x_new)
        return mu[0, 0], var[0, 0]
    else:
        mu, _ = model.predict(x_new)
        return mu[0, 0]

data_train = pd.read_csv('../../dataset/training.csv')
data_test = pd.read_csv('../../dataset/testing_validation.csv')
column_data = list(data_train.columns)

# 数据处理，类别数据

data_train['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_train['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

data_test['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_test['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

column = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
             'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',  'Tdewpoint', 'NSM',  'Day_of_week']

# for index in column:
#     scaler = MinMaxScaler()
#     data_train[index] = scaler.fit_transform((data_train[index].values).reshape(-1, 1))
# for index in column:
#     scaler = MinMaxScaler()
#     data_test[index] = scaler.fit_transform((data_test[index].values).reshape(-1, 1))

column_LM = ['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
             'T_out', 'RH_out',  'Day_of_week', 'NSM']
x_train = data_train[column_LM].values
x_test = data_test[column_LM].values

y_train = data_train['Appliances'].values
y_test = data_test['Appliances'].values
m, n = np.shape(x_train)
print('原训练集规模', np.shape(x_train))  # 14803


a_gpr = A_GPR()

'''

_, x_train_l,  _,  y_train_l = train_test_split(x_train, y_train, test_size=200/m)
print('实际应用于构建low fidelity GP 的数据数目：', np.shape(x_train_l)[0])
# 为 low fidelity 数据添加噪声
mu_l = 0
sigma_l = 0.5
for i in range(np.shape(x_train_l)[0]):
    y_train_l[i] = y_train_l[i] * 1.2 - 1
list_mean_mse_gpr = []
list_mean_mse_lgb = []
list_mean_mse_svr = []
list_average = []
plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(121)
ax_2 = plt.subplot(122)
left = 1000
right = 1001
dist = 1
init_size = 2
for it in list(range(left, right, dist)):
    print('训练数据样本数目：', it)
    print('候选数据集大小：', np.shape(x_train))
    if it > np.shape(x_train)[0]:
        print('抽样样本点大于数据集本身')
        break
    list_mse_gpr = []
    list_mse_lgb = []
    list_mse_svr = []
    # list_mse_con = []
    list_w = []
    for i in range(1):
        temp_iter = it - init_size
        # 初始化GP的训练数据
        x_init, y_init = a_gpr.sample_point(x_data_conda=np.array(x_train, ndmin=2), y_data_conda=np.array(y_train).reshape(-1, 1), iter=init_size)
        hf_gp, list_w_hf, x_gp, y_gp = a_gpr.creat_gp_model(max_loop=temp_iter, x_init_l=x_train_l, y_init_l=np.array(y_train_l).reshape(-1, 1), x_init_h=x_init, y_init_h=y_init,
                                                n_start=1, n_single=500, x_conda=np.array(x_train, ndmin=2), y_conda=np.array(y_train).reshape(-1, 1)
                                                        )
        list_w.append(list_w_hf)
        y_pre_gpr = [a_gpr.predict_mu_var(np.reshape(x_test[k], (1, -1)), hf_gp, re_var=False) for k in range(np.shape(x_test)[0])]
        # lgm 模型
        x_train_m, y_train_m = a_gpr.sample_point(x_train, y_train, iter=it)
        model_lgb = lgb.LGBMRegressor()
        model_lgb.fit(x_train_m, np.reshape(y_train_m, (1, -1))[0])
        y_pre_lgb = [model_lgb.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]
        # svm 模型
        x_train_r, y_train_r = a_gpr.sample_point(x_train, y_train, iter=it)
        model_svr = SVR()
        model_svr.fit(x_train_r, np.reshape(y_train_r, (1, -1))[0])
        y_pre_svr = [model_svr.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]
        # 高斯模型
        # k_rbf = RBF(input_dim=n, variance=1, lengthscale=0.5)
        # gp_model = GPRegression(x_train_m, np.reshape(y_train_m, (-1, 1)), kernel=k_rbf)
        # gp_model.optimize(messages=False)
        # y_pre_con = [pre_gp_mu_var(np.reshape(x_test[i], (1, -1)), gp_model) for i in range(np.shape(x_test)[0])]
        list_mse_gpr.append(np.sqrt(mean_squared_error(y_test, y_pre_gpr)))
        list_mse_lgb.append(np.sqrt(mean_squared_error(y_test, y_pre_lgb)))
        list_mse_svr.append(np.sqrt(mean_squared_error(y_test, y_pre_svr)))
        # list_mse_con.append(np.sqrt(mean_squared_error(y_test, y_pre_con)))
    list_mean_mse_gpr.append(np.mean(list_mse_gpr))
    list_mean_mse_lgb.append(np.mean(list_mse_lgb))
    list_mean_mse_svr.append(np.mean(list_mse_svr))
    # list_mean_mse_con.append(np.mean(list_mse_con))
    list_average = np.mean(list_w, axis=0)
    plt.sca(ax_1)
    plt.plot(list_average, lw=1.5, label='%s-st' % str(it))
plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('w_hf')
plt.xlabel('iter')
plt.title('energy predict')
plt.sca(ax_2)
plt.plot(list(range(left, right, dist)), list_mean_mse_gpr, lw=1.5, label='gpr_m')
plt.plot(list(range(left, right, dist)), list_mean_mse_lgb, lw=1.5, label='lgb_m')
plt.plot(list(range(left, right, dist)), list_mean_mse_svr, lw=1.5, label='svr_m')
# plt.plot(list(range(left, right, dist)), list_mean_mse_con, lw=1.5, label='con_m')
plt.axis('tight')
plt.legend(loc=0)  # 图例位置自动
plt.ylabel('RMSE')
plt.xlabel('iter')
plt.title('energy predict')
print('gpr', list_mean_mse_gpr)
print('lgb', list_mean_mse_lgb)
print('svr', list_mean_mse_svr)
# print('con', list_mean_mse_con)
plt.show()


'''

def pre_gp_mu_var(x_new, model, return_var=False):
    if return_var:
        mu, var = model.predict(x_new)
        return mu[0, 0], var[0, 0]
    else:
        mu, _ = model.predict(x_new)
        return mu[0, 0]



x_train_m, y_train_m = a_gpr.sample_point(x_train, y_train, iter=1000)


k_rbf = RBF(input_dim=n, variance=0.5, lengthscale=1)
gp_model = GPRegression(x_train_m, np.reshape(y_train_m, (-1, 1)), kernel=k_rbf)
gp_model.optimize(messages=False)

model_lgb = lgb.LGBMRegressor()
model_lgb.fit(x_train_m, np.reshape(y_train_m, (1, -1))[0])

model_svr = SVR()
model_svr.fit(x_train_m,np.reshape(y_train_m, (1, -1))[0])


y_pre_con = [pre_gp_mu_var(np.reshape(x_test[i], (1, -1)), gp_model) for i in range(np.shape(x_test)[0])]
y_pre_lgb = [model_lgb.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]
y_pre_svr = [model_svr.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]
print(y_pre_con)
print('mse_con:', np.sqrt(mean_squared_error(y_test, y_pre_con)))
print('mse_lgb:', np.sqrt(mean_squared_error(y_test, y_pre_lgb)))
print('mse_svr:', np.sqrt(mean_squared_error(y_test, y_pre_svr)))
