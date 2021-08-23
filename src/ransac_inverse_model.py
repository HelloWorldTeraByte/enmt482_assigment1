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

ransac = RANSACRegressor(IrModel([1, 1]), random_state=1, min_samples=5)
print(np.shape(raw_ir3))
print(np.shape(raw_ir3.reshape(-1,1)))

ransac.fit(raw_ir3.reshape(-1,1), distance)
#ransac.fit(distance.reshape(-1,1), raw_ir2)

print(ransac.estimator_.coeffs)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

fig, axes = plt.subplots(2, 3)
fig.suptitle('Calibration data')

axes[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
axes[0, 0].set_title('IR1')

axes[0, 1].plot(distance, raw_ir2, '.', alpha=0.2)
axes[0, 1].plot(distance, ransac.predict(distance.reshape(-1,1)))
axes[0, 1].set_title('IR2')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
#axes[1, 1].plot(distance, ransac.predict(distance))
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')

plt.show()


'''
from matplotlib import pyplot as plt
import numpy as np

from sklearn.linear_model import (LinearRegression, 
                                  TheilSenRegressor, 
                                  RANSACRegressor, 
                                  HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

#Train data
X = np.random.normal(size=400)
y = np.sin(X)
X = X[:, np.newaxis]


# Test Data
X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

plt.scatter(X,y)

# Small outliers on Y
y_errors = y.copy()
y_errors[::3] = 3

#Small outliers on X
X_errors = X.copy()
X_errors[::3] = 3

#Small outliers on Y
y_errors_large = y.copy()
y_errors_large[::3] = 10

#Large outliers on X
X_errors_large = X.copy()
X_errors_large[::3] = 10

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()
ax[0, 0].scatter(X,y_errors)
ax[0, 0].set_title("Small Y outliers")

ax[0, 1].scatter(X_errors,y)
ax[0, 1].set_title("Small X outliers")

ax[1, 0].scatter(X,y_errors_large)
ax[1, 0].set_title("Large Y outliers")

ax[1, 1].scatter(X_errors_large,y)
ax[1, 1].set_title("Large X outliers")

estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor())]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
lw = 3
x_plot = np.linspace(X.min(), X.max())

for title, this_X, this_y in [
        ('Modeling Errors Only', X, y),
        ('Corrupt X, Small Deviants', X_errors, y),
        ('Corrupt y, Small Deviants', X, y_errors),
        ('Corrupt X, Large Deviants', X_errors_large, y),
        ('Corrupt y, Large Deviants', X, y_errors_large)]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, 'b+')

    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(3), estimator)
        model.fit(this_X, this_y)
        mse = mean_squared_error(model.predict(X_test), y_test)
        y_plot = model.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                 linewidth=lw, label='%s: error = %.3f' % (name, mse))

    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()
'''