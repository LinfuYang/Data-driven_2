
import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from sklearn.linear_model import Ridge
from pyGPGO.energy_data import load_energy
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':

    # ['date', 'Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
    # 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2', 'NSM', 'WeekStatus', 'Day_of_week']

    x_train, x_test, y_train, y_test = load_energy()
    def f(x):
        return -np.mean(np.sqrt(-(cross_val_score(Ridge(alpha=x), x_train, y=y_train, scoring='neg_mean_squared_error', cv=10))))

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {'x': ('cont', [0, 100])}


    gpgo = GPGO(gp, acq, f, param)
    gpgo.run(max_iter=100)
    best_x, best_y = gpgo.getResult()
    print('best_x:', best_x)
    print('best_y:', best_y)
    model_Ridge = Ridge(alpha=best_x)

    model_Ridge.fit(x_train, y_train)
    y_predict = model_Ridge.predict(x_test)

    RMSE_Ridge = np.sqrt(mean_squared_error(y_test, y_predict))
    R2_Ridge = r2_score(y_test, y_predict)
    MAE_Ridge = median_absolute_error(y_test, y_predict)

    print('****************'+'Ridge'+'****************')
    print('RMSE_Ridge:', RMSE_Ridge)
    print('R2_Ridge:', R2_Ridge)
    print('MAE_Ridge:', MAE_Ridge)




