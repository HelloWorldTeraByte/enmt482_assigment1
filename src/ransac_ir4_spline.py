import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.interpolate import splev, splrep
from sklearn.linear_model import RANSACRegressor
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time_data, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

class SplineRegressor(object):
    def __init__(self, coeffs=None):
        self.coeffs = coeffs
        self.smooth = 200

    def fit(self, X, y):
        self.distance_sorted, self.measurement_sorted = zip(*sorted(zip(X.ravel(), y)))
        self.spline = splrep(self.distance_sorted, self.measurement_sorted, s=self.smooth)

        #self.spline_meas = splev(self.distance, self.spline)

        #popt, pcov = curve_fit(self.model, X.ravel(), y)
        self.coeffs = self.spline

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        self.spline_meas = splev(X.ravel(), self.spline)
        return self.spline_meas

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

class IR4SensorModel(object):
    def __init__(self, distance, measurement, smooth=130):
        self.distance = distance
        self.measurement = measurement
        self.smooth_val = smooth

        self.plots_init()
        self.calculate()
        self.plots_draw()
    
    def slider_update(self, val):
        self.smooth_val = self.smooth_slider.val
        self.calculate()
        self.plots_draw()

    def calculate(self):
        #self.distance_sorted, self.measurement_sorted = zip(*sorted(zip(self.distance, self.measurement)))
        #self.spline = splrep(self.distance_sorted, self.measurement_sorted, s=self.smooth_val)
        #print(self.spline)

        ##self.spline_dist = np.linspace(0.05, 3.5, 500)
        #self.spline_meas = splev(self.distance, self.spline)

        self.ransac = RANSACRegressor(SplineRegressor(), random_state=0, min_samples=10,residual_threshold=(median_abs_deviation(self.measurement)+self.smooth_val))
        self.ransac.fit(self.distance.reshape(-1,1), self.measurement)

        self.inlier_mask = self.ransac.inlier_mask_
        self.outlier_mask = np.logical_not(self.inlier_mask)

        self.meas_inliers = np.delete(self.measurement, self.outlier_mask)
        self.dist_inliers = np.delete(self.distance, self.outlier_mask)

        self.ransac_pred = self.ransac.predict(self.distance.reshape(-1,1))

        self.errors = self.measurement - self.ransac_pred
        self.error_var = np.var(self.errors)
        print('Error Variance: ', self.error_var)

    def plots_init(self):
        self.liers_fig, self.liers_ax = plt.subplots()
        plt.title("Inliers and Outliers")

        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.sensor_fig, self.sensor_ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        plt.title("IR4")
        self.sens_ax_smooth = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.smooth_slider = Slider(self.sens_ax_smooth, 'Smoothing', 0, 1, self.smooth_val)
        self.smooth_slider.on_changed(self.slider_update)

    def plots_draw(self):
        self.liers_ax.clear()
        self.liers_ax.scatter(self.distance, self.measurement)
        self.liers_ax.scatter(self.dist_inliers, self.meas_inliers)
        self.liers_fig.canvas.draw()

        self.err_ax[0].clear()
        self.err_ax[1].clear()
        self.err_ax[0].scatter(self.distance, self.errors)
        self.err_ax[1].hist(self.errors, 100)
        self.err_fig.canvas.draw()

        self.sensor_ax.clear()
        self.sensor_ax.plot(self.distance, self.measurement, '.')
        self.sensor_ax.plot(self.distance, self.ransac_pred, color='red', linewidth=2)
        self.sensor_fig.canvas.draw()

ir4_sen = IR4SensorModel(distance, raw_ir4)
plt.show()
