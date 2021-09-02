################################################################################
#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,                 #
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/                  #
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~                #
################################################################################
#                                                                              #
#                                  ir3.py                                      #
#                                                                              #
################################################################################
# Authors:        Jason Ui
#                 Randipa Gunathilake
#
# Date created:       22/08/2021
# Date Last Modified: 31/08/2021
################################################################################
#  Module Description:
#
#
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate.fitpack import spalde
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splev, splrep
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_squared_error

ir3_smooth_val = 20

class Ir3Regressor(object):
    def __init__(self, coeffs=None):
        self.coeffs = coeffs

    def fit(self, X, y):
        self.distance_sorted, self.measurement_sorted = zip(*sorted(zip(X.ravel(), y)))
        self.spline = splrep(self.distance_sorted, self.measurement_sorted, s = ir3_smooth_val)
        self.coeffs = self.spline

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        return splev(X.ravel(), self.spline)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

class Ir3Sensor(object):
    def __init__(self, distance, measurement, should_plot = False):
        self.distance = distance
        self.measurement = measurement
        self.ransac = RANSACRegressor(Ir3Regressor(), random_state=0, min_samples=1000,residual_threshold=(median_abs_deviation(self.measurement)+0.5))

        self.ransac.fit(self.distance.reshape(-1,1), self.measurement)
        self.inlier_mask = self.ransac.inlier_mask_
        self.outlier_mask = np.logical_not(self.inlier_mask)

        self.meas_inliers = np.delete(self.measurement, self.outlier_mask)
        self.dist_inliers = np.delete(self.distance, self.outlier_mask)

        self.ransac_pred = self.ransac.predict(self.dist_inliers.reshape(-1,1))
        self.errors = self.meas_inliers - self.ransac_pred
        self.error_var = np.var(self.errors)

        self.spline = self.ransac.estimator_.spline

        self.func_offset = lambda x, a : splev(x, self.spline) - a

        self.inv_x = []
        self.inv_y = []

        for y in np.linspace(0.5, 3, 1000):
            try:
                x = optimize.newton(self.func_offset, 0.25, args=(y,))
                self.inv_x = np.append(self.inv_x, x)
                self.inv_y = np.append(self.inv_y, y)
            except:
                pass

        self.inv_spline = splrep(self.inv_y, self.inv_x)

        self.inv_pred_y = np.linspace(0.25, 3, 100)
        self.inv_pred_x = splev(self.inv_pred_y, self.inv_spline)

        self.err_norm_inv_x = np.linspace(0.25, 3, 1000)
        self.err_norm_inv = np.zeros(1000)

        for i in range(np.size(self.err_norm_inv_x)):
            normal_pred = self.ransac.predict(self.err_norm_inv_x[i].reshape(-1,1)) 
            self.err_norm_inv[i] = self.err_norm_inv_x[i] - splev(normal_pred, self.inv_spline)

        if(should_plot):
            self.plots_init()
            self.plots_draw()

    # Linearizing about x_0
    def x_est_mle(self, z, x_0):
        h_x0 = splev(x_0, self.spline)
        h_derv_x0 = spalde(x_0, self.spline)[1]
        c = h_derv_x0
        d = h_x0 - x_0 * h_derv_x0

        self.test_x = np.linspace(0, 1, 100)
        self.test_y = c * self.test_x + d

        x_est = (z - h_x0)/h_derv_x0 + x_0

        return x_est
    
    def plots_init(self):
        self.liers_fig, self.liers_ax = plt.subplots()
        plt.title("Inliers and Outliers")

        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.sensor_fig, self.sensor_ax = plt.subplots()
        plt.title("IR3")

        self.inv_model_fig, self.inv_model_ax = plt.subplots()
        plt.title("Inverse model")

        self.inv_model_err_fig, self.inv_model_err_ax = plt.subplots()
        plt.title("Inverse model errors")

    def plots_draw(self):
        self.liers_ax.scatter(self.distance, self.measurement)
        self.liers_ax.scatter(self.dist_inliers, self.meas_inliers)

        self.err_ax[0].scatter(self.dist_inliers, self.errors)
        self.err_ax[1].hist(self.errors, 100)

        self.sensor_ax.plot(self.distance, self.measurement, '.')
        self.sensor_ax.plot(self.test_x, self.test_y)
        self.sensor_ax.plot(self.dist_inliers, self.ransac_pred, color='red', linewidth=2)

        self.inv_model_ax.plot(self.inv_pred_y, self.inv_pred_x)

        self.inv_model_err_ax.plot(self.err_norm_inv_x, self.err_norm_inv)


if __name__ == "__main__":
    filename = '../data/calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time_data, distance, velocity_command, \
        raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

    ir3_sen = Ir3Sensor(distance, raw_ir3, should_plot=False)
    print(ir3_sen.x_est_mle(0.8, 0.4))
    print("Error Variance: ", ir3_sen.error_var)
    ir3_sen.plots_init()
    ir3_sen.plots_draw()

    plt.show()