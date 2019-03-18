from sklearn.model_selection import train_test_split
import numpy as np
from coding.ABO.AGPR import A_GPR
import matplotlib.pyplot as plt
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')
a_gpr = A_GPR()

data = pd.read_excel('../../dataset/cccp_data.xlsx')
# ['AT', 'V', 'AP', 'RH', 'PE']
x_data = data[['AT', 'V', 'AP', 'RH']].values
y_data = data[['PE']].values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.8)

m = len(y_train)
plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(121)
ax_2 = plt.subplot(122)
list_average = list(range(m))
plt.sca(ax_1)
mu = 0
sigma=30
plt.plot(list_average, y_train, lw=1.5, label='y_train')
plt.sca(ax_2)
y_train_1 = [y_train[i] + random.gauss(mu, sigma) for i in range(m)]
plt.plot(list_average, y_train_1, lw=1.5, label='y_train')
plt.show()
