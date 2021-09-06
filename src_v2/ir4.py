################################################################################
#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,                 #
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/                  #
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~                #
################################################################################
#                                                                              #
#                                     ir4.py                                   #
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
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splev, splrep
from scipy.interpolate.fitpack import spalde
from sklearn.metrics import mean_squared_error

from util import find_nearest_index

ir4_smooth_val = 50

class Ir4Regressor(object):
    def __init__(self, coeffs=None):
        self.coeffs = coeffs

    def fit(self, X, y):
        self.distance_sorted, self.measurement_sorted = zip(*sorted(zip(X.ravel(), y)))
        self.spline = splrep(self.distance_sorted, self.measurement_sorted, s = ir4_smooth_val)
        self.coeffs = self.spline

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        return splev(X.ravel(), self.spline)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

class Ir4Sensor(object):
    def __init__(self, distance, measurement, should_plot = False):
        self.distance = distance
        self.measurement = measurement
        self.dist_min = np.min(self.distance)
        self.dist_max = np.max(self.distance)

        self.ransac = RANSACRegressor(Ir4Regressor(), random_state=0, min_samples=1000)

        self.ransac.fit(self.distance.reshape(-1,1), self.measurement)
        self.inlier_mask = self.ransac.inlier_mask_
        self.outlier_mask = np.logical_not(self.inlier_mask)

        self.meas_inliers = np.delete(self.measurement, self.outlier_mask)
        self.dist_inliers = np.delete(self.distance, self.outlier_mask)

        self.ransac_pred = self.ransac.predict(self.dist_inliers.reshape(-1,1))
        self.errors = self.meas_inliers - self.ransac_pred
        self.error_var = np.var(self.errors)

        bin_dist = 0.1
        s = 0
        e = s + bin_dist
        s_ind = find_nearest_index(self.distance, s)
        
        bin_num = int(np.round(self.dist_max/bin_dist))
        self.bin_err_var = np.zeros(bin_num)
        self.bin_err_var_x = np.linspace(self.dist_min, self.dist_max, bin_num)

        for i in range(0, bin_num):
            if(i == bin_num - 1):
                e_ind = -1
            else:
                e_ind = find_nearest_index(self.distance, e)
            bin_pred = self.ransac.predict(self.distance[s_ind:e_ind].reshape(-1,1))
            bin_err = self.measurement[s_ind:e_ind] - bin_pred
            self.bin_err_var[i] = np.var(bin_err)
            s = e
            e = e + bin_dist

        self.err_spline_x = np.linspace(self.dist_min, self.dist_max, 100)
        self.err_spline = splrep(self.bin_err_var_x, self.bin_err_var)
        self.err_spline_y = splev(self.err_spline_x, self.err_spline)

        self.spline = self.ransac.estimator_.spline

        self.dist_min = np.min(self.distance)
        self.dist_max = np.max(self.distance)

        if(should_plot):
            self.plots_init()
            self.plots_draw()

    # Linearizing about x_0
    def x_est_mle(self, z, x_0):
        if(x_0 < self.dist_min):
            x_0 = self.dist_min
        if(x_0 > self.dist_max):
            x_0 = self.dist_max

        h_x0 = splev(x_0, self.spline)
        h_derv_x0 = spalde(x_0, self.spline)[1]
        c = h_derv_x0
        d = h_x0 - x_0 * h_derv_x0

        self.test_x = np.linspace(0, 1, 100)
        self.test_y = c * self.test_x + d

        x_est = (z - h_x0)/h_derv_x0 + x_0

        return x_est

    def var_estimator_at_x0(self, x_0):
        if(x_0 < self.dist_min):
            x_0 = self.dist_min
        if(x_0 > self.dist_max):
            x_0 = self.dist_max

        h_x0 = splev(x_0, self.spline)
        h_derv_x0 = spalde(x_0, self.spline)[1]

        c = h_derv_x0
        d = h_x0 - x_0 * h_derv_x0

        var_sen = splev(x_0, self.err_spline)

        var = var_sen / (c ** 2)

        return var

        if(x_0 < 1):
            var = 10

 
    def plots_init(self):
        self.liers_fig, self.liers_ax = plt.subplots()
        plt.title("Inliers and Outliers")

        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.sensor_fig, self.sensor_ax = plt.subplots()
        plt.title("IR4")

        self.bin_err, self.bin_err = plt.subplots()
        plt.title("Bin Errors")

    def plots_draw(self):
        self.liers_ax.scatter(self.distance, self.measurement)
        self.liers_ax.scatter(self.dist_inliers, self.meas_inliers)

        self.err_ax[0].scatter(self.dist_inliers, self.errors)
        self.err_ax[1].hist(self.errors, 100)

        self.sensor_ax.plot(self.distance, self.measurement, '.')
        self.sensor_ax.plot(self.dist_inliers, self.ransac_pred, color='red', linewidth=2)

        self.bin_err.plot(self.bin_err_var_x, self.bin_err_var)
        self.bin_err.plot(self.err_spline_x, self.err_spline_y)

if __name__ == "__main__":
    filename = '/home/helloworldterabyte/projects/enmt482-2021_robotic_assignment/data/calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time_data, distance, velocity_command, \
        raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

    ir4_sen = Ir4Sensor(distance, raw_ir4, should_plot=True)
    print("Error Variance: ", ir4_sen.error_var)

    plt.show()