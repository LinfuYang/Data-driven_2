
import numpy as np
from sklearn.svm import SVR
from pyGPGO.energy_data import load_energy
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':

    # ['date', 'Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
    # 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2', 'NSM', 'WeekStatus', 'Day_of_week']

    x_train, x_test, y_train, y_test = load_energy()
    print(x_train.columns)

    model_base = SVR(C=12, gamma=3.125)

    model_base.fit(x_train, y_train)
    y_predict = model_base.predict(x_test)

    RMSE_SVR = np.sqrt(mean_squared_error(y_test, y_predict))
    R2_SVR = r2_score(y_test, y_predict)
    MAE_SVR = median_absolute_error(y_test, y_predict)

    print('****************'+'SVR'+'****************')
    print('RMSE_SVR:', RMSE_SVR)
    print('R2_SVR:', R2_SVR)
    print('MAE_SVR:', MAE_SVR)




