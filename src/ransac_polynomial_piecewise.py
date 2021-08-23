import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

brkpt = find_nearest_index(distance, 0.4)
print(brkpt)

model_lin_x = distance[0:brkpt]
model_lin_y = raw_ir3[0:brkpt]

model_pol_x = distance
model_pol_y = raw_ir3

ransac_lin = RANSACRegressor()
ransac_lin.fit(model_lin_x.reshape(-1,1), model_lin_y)
pred_lin = ransac_lin.predict(model_lin_x.reshape(-1,1))

ransac_pol = make_pipeline(PolynomialFeatures(3), RANSACRegressor())
ransac_pol.fit(model_pol_x.reshape(-1,1), model_pol_y)
#mse = mean_squared_error(ransac.predict(distance.reshape(-1,1)), raw_ir3)
pred_pol = ransac_pol.predict(model_pol_x.reshape(-1,1))

fig, axes = plt.subplots(2, 3)
fig.suptitle('Calibration data')

axes[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
axes[0, 0].set_title('IR1')

axes[0, 1].plot(distance, raw_ir2, '.', alpha=0.2)
axes[0, 1].set_title('IR2')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].plot(model_lin_x, pred_lin)
axes[0, 2].plot(model_pol_x, pred_pol)
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')


plt.show()