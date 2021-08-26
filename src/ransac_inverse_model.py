import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

class IrModel(object):
    def __init__(self, coeffs=None):
        self.coeffs = coeffs

    def model(self, x, k1, k2):
        return k1 / (k2 + x)

    def fit(self, X, y):
        popt, pcov = curve_fit(self.model, X.ravel(), y)
        self.coeffs = popt

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        return self.model(X.ravel(), self.coeffs[0], self.coeffs[1])

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

ransac = RANSACRegressor(IrModel([1, 1]), random_state=0)

ransac.fit(raw_ir2.reshape(-1,1), distance)
#ransac.fit(distance.reshape(-1,1), raw_ir2)

print(ransac.estimator_.coeffs)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
meas_inliers = np.delete(raw_ir2, outlier_mask)
dist_inliers = np.delete(distance, outlier_mask)

fig, axes = plt.subplots(2, 3)
fig.suptitle('Calibration data')

axes[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
axes[0, 0].set_title('IR1')

axes[0, 1].plot(distance, raw_ir2, '.')
axes[0, 1].plot(distance, ransac.predict(distance.reshape(-1,1)))
#axes[0, 1].scatter(dist_inliers, meas_inliers, linewidth=2, color='red')
axes[0, 1].set_title('IR2')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')


plt.figure()
plt.scatter(dist_inliers, meas_inliers, linewidth=2, color='red')

plt.show()