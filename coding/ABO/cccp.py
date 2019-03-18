from sklearn.datasets import load_boston
import pandas as pd
from GPy.models import GPRegression
from GPy.kern import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import lightgbm as lgb
import numpy as np
import random
from coding.ABO.AGPR import A_GPR
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
a_gpr = A_GPR()

data = pd.read_excel('../../dataset/cccp_data.xlsx')
# ['AT', 'V', 'AP', 'RH', 'PE']
x_data = data[['AT', 'V', 'AP', 'RH']].values
y_data = data[['PE']].values

# for index in range(np.shape(x_data)[0]):
#     scaler = StandardScaler()
#     x_data[index] = scaler.fit_transform(np.array(x_data[index])[0].reshape(-1, 1))
#
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

m, n = np.shape(x_train)
print('训练数据总量', m)

_, x_train_l, _, y_train_l = train_test_split(x_train, y_train, test_size=60/m)

# x_train_l, y_train_l = a_gpr.sample_point(x_train, y_train, iter=100)

print('低似真度数据样本量：', np.shape(x_train_l)[0])

# 为 low fidelity 数据添加噪声
mu = 0
sigma = 30
for i in range(np.shape(x_train_l)[0]):
    y_train_l[i] = y_train_l[i] + random.gauss(0, sigma)
    # y_train_l[i] = y_train_l[i] * 1.1 - 10

x_train_l = np.array(x_train_l, ndmin=2)
y_train_l = np.reshape(y_train_l, (-1, 1))

list_mean_mse_gpr = []
list_mean_mse_lgb = []
list_mean_mse_svr = []

list_average = []
plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(121)
ax_2 = plt.subplot(122)
left = 100
right = 201
dist = 50
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
    list_w = []

    for i in range(5):
        temp_iter = it - init_size
        # 初始化GP的训练数据
        single = 20
        hf_gp, list_w_hf, x_gp, y_gp = a_gpr.creat_gp_model(max_loop=temp_iter, x_init_l=x_train_l, y_init_l=y_train_l,init_num=init_size,
                                                n_start=1, n_single=single, x_conda=np.array(x_train, ndmin=2), y_conda=np.array(y_train).reshape(-1, 1)
                                                            )
        list_w.append(list_w_hf)
        y_pre_gpr = [a_gpr.predict_mu_var(np.reshape(x_test[k], (1, -1)), hf_gp, re_var=False) for k in range(np.shape(x_test)[0])]

        x_train_m, y_train_m = a_gpr.sample_point(x_train, y_train, iter=it, is_init=True)
        # lgm 模型
        # _, x_train_m, _, y_train_m = train_test_split(x_train, y_train, test_size=(it / m))
        model_lgb = lgb.LGBMRegressor()
        model_lgb.fit(x_train_m, np.reshape(y_train_m, (1, -1))[0])
        y_pre_lgb = [model_lgb.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]

        # svm 模型
        x_train_r, y_train_r = a_gpr.sample_point(x_train, y_train, iter=it, is_init=True)
        # _, x_train_r, _, y_train_r = train_test_split(x_train, y_train, test_size=(it / m))

        model_svr = SVR()
        model_svr.fit(x_train_r, np.reshape(y_train_r, (1, -1))[0])
        y_pre_svr = [model_svr.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]

        list_mse_gpr.append(mean_squared_error(y_test, y_pre_gpr))
        list_mse_lgb.append(mean_squared_error(y_test, y_pre_lgb))
        list_mse_svr.append(mean_squared_error(y_test, y_pre_svr))

    list_mean_mse_gpr.append(np.mean(list_mse_gpr))
    list_mean_mse_lgb.append(np.mean(list_mse_lgb))
    list_mean_mse_svr.append(np.mean(list_mse_svr))

    list_average = np.mean(list_w, axis=0)
    plt.sca(ax_1)
    plt.plot(list_average, lw=1.5, label='%s-st' % str(it))


plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('w_hf')
plt.xlabel('iter')
plt.title('boston_price')

plt.sca(ax_2)
plt.plot(list(range(left, right, dist)), list_mean_mse_lgb, lw=1.5, label='lgb_m')
plt.plot(list(range(left, right, dist)), list_mean_mse_gpr, lw=1.5, label='gpr_m')
plt.plot(list(range(left, right, dist)), list_mean_mse_svr, lw=1.5, label='svr_m')
plt.axis('tight')
plt.legend(loc=0)  # 图例位置自动
plt.ylabel('MSE')
plt.xlabel('iter')
plt.title('boston_price')


print('gpm', list_mean_mse_gpr)
print('lgb', list_mean_mse_lgb)
print('svr', list_mean_mse_svr)

plt.show()

'''


def pre_gp_mu_var(x_new, model, return_var=False):
    if return_var:
        mu, var = model.predict(x_new)
        return mu[0, 0], var[0, 0]
    else:
        mu, _ = model.predict(x_new)
        return mu[0, 0]

x_train_r, y_train_r = a_gpr.sample_point(x_train, y_train, iter=200, is_init=True)

k_rbf = RBF(input_dim=n, variance=0.5, lengthscale=1)
gp_model = GPRegression(x_train_r, np.reshape(y_train_r, (-1, 1)), kernel=k_rbf)
gp_model.optimize(messages=False)

x_train_m, y_train_m = a_gpr.sample_point(x_train, y_train, iter=200, is_init=True)

model_lgb = lgb.LGBMRegressor()
model_lgb.fit(x_train_r, np.reshape(y_train_r, (1, -1))[0])

x_train_p, y_train_p = a_gpr.sample_point(x_train_m, y_train_m, iter=200, is_init=True)

model_svr = SVR()
model_svr.fit(x_train_p, np.reshape(y_train_p, (1, -1))[0])


y_pre_con = [pre_gp_mu_var(np.reshape(x_test[i], (1, -1)), gp_model) for i in range(np.shape(x_test)[0])]
y_pre_lgb = [model_lgb.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]
y_pre_svr = [model_svr.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]
print(y_pre_con)
print('mse_con:', mean_squared_error(y_test, y_pre_con))
print('mse_lgb:', mean_squared_error(y_test, y_pre_lgb))
print('mse_svr:', mean_squared_error(y_test, y_pre_svr))



'''
