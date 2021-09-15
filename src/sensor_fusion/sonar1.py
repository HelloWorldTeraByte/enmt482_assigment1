################################################################################
#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,                 #
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/                  #
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~                #
################################################################################
#                                                                              #
#                                  sonar1.py                                   #
#                                                                              #
################################################################################
# Authors:        Jason Ui
#                 Randipa Gunathilake
#
# Date created:       20/08/2021
# Date Last Modified: 31/08/2021
################################################################################
#  Module Description:
#
#
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splrep, splev
from scipy.stats import median_abs_deviation
import pickle
import os
import errno

from utils import find_nearest_index

class Sonar1Sensor(object):
    def __init__(self, distance=[], measurement=[], use_saved=True, should_plot=False):
        try:
            os.mkdir('data')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.pickle_loc = 'data/sonar1.pckl'

        if(use_saved):
            with open(self.pickle_loc, "rb") as f:
                self.model_c, self.model_d, self.err_spline = pickle.load(f)
                return

        self.distance = distance
        self.measurement = measurement
        self.dist_min = np.min(self.distance)
        self.dist_max = np.max(self.distance)
        self.ransac = RANSACRegressor(random_state=0, residual_threshold=(
            median_abs_deviation(self.measurement))-0.75)

        self.ransac.fit(self.distance.reshape(-1, 1), self.measurement)
        self.inlier_mask = self.ransac.inlier_mask_
        self.outlier_mask = np.logical_not(self.inlier_mask)

        self.meas_inliers = np.delete(self.measurement, self.outlier_mask)
        self.dist_inliers = np.delete(self.distance, self.outlier_mask)

        self.meas_outliers = np.delete(self.measurement, self.inlier_mask)
        self.dist_outliers = np.delete(self.distance, self.inlier_mask)

        self.model_pred_inliers = self.ransac.predict(
            self.dist_inliers.reshape(-1, 1))
        self.model_err_inliers = self.meas_inliers - self.model_pred_inliers
        #self.error_var = np.var(self.model_err_inlier)

        self.model_pred = self.ransac.predict(self.distance.reshape(-1, 1))
        self.model_err = self.measurement - self.model_pred
        self.model_err_var = np.var(self.model_err)
        self.model_error_var = self.model_err_var

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
            bin_pred = self.ransac.predict(
                self.distance[s_ind:e_ind].reshape(-1, 1))
            bin_err = self.measurement[s_ind:e_ind] - bin_pred
            self.bin_err_var[i] = np.var(bin_err)
            s = e
            e = e + bin_dist
            s_ind = find_nearest_index(self.distance, s)

        self.err_spline_x = np.linspace(self.dist_min, self.dist_max, 100)
        self.err_spline = splrep(self.bin_err_var_x, self.bin_err_var)
        self.err_spline_y = splev(self.err_spline_x, self.err_spline)

        self.model_d = self.ransac.estimator_.intercept_
        self.model_c = self.ransac.estimator_.coef_[0]

        pickle_data = [self.model_c, self.model_d, self.err_spline]
        with open(self.pickle_loc, "wb") as f:
            pickle.dump(pickle_data, f)

        if(should_plot):
            self.plots_init()
            self.plots_draw()

    def x_est_mle(self, z):
        x_est = (z - self.model_d) / self.model_c
        return x_est

    def var_estimator(self, x):
        #var = self.error_var / (self.ransac.estimator_.coef_[0] ** 2)
        var = splev(x, self.err_spline) / (self.model_c ** 2)
        return var

    def plots_init(self):
        self.liers_fig, self.liers_ax = plt.subplots()

        self.err_fig, self.err_ax = plt.subplots(2)

        self.sensor_fig, self.sensor_ax = plt.subplots()

        self.bin_err, self.bin_err = plt.subplots()
        plt.title("Bin Errors")

    def plots_draw(self):
        self.liers_ax.scatter(self.dist_outliers,
                              self.meas_outliers, s=10, label='Outliers')
        self.liers_ax.scatter(
            self.dist_inliers, self.meas_inliers, color='orange', s=10, label='Inliers')
        self.liers_ax.set_title('Inliers and Outliers of Sonar 1 Sensor')
        self.liers_ax.set_xlabel('Distance')
        self.liers_ax.set_ylabel('Measurement')
        self.liers_ax.legend(loc='upper left')

        self.err_ax[0].scatter(self.dist_inliers, self.model_err_inliers)
        self.err_ax[1].hist(self.model_err_inliers, 100)

        self.err_ax[0].set_title('Errors for Distance')
        self.err_ax[0].set_xlabel('Distance')
        self.err_ax[0].set_ylabel('Error')
        self.err_ax[1].set_title('Histogram of Errors')
        self.err_ax[1].set_xlabel('Error')
        self.err_ax[1].set_ylabel('Frequency')

        self.sensor_ax.plot(self.distance, self.measurement,
                            '.', label='Measurements')
        self.sensor_ax.plot(self.dist_inliers, self.model_pred_inliers,
                            color='red', linewidth=2, label='Model')
        self.sensor_ax.set_title('Sonar 1 Model')
        self.sensor_ax.set_xlabel('Distance')
        self.sensor_ax.set_ylabel('Measurement')
        self.sensor_ax.legend(loc='upper left')

        self.bin_err.plot(self.bin_err_var_x, self.bin_err_var)
        self.bin_err.plot(self.err_spline_x, self.err_spline_y)


if __name__ == "__main__":
    filename = '../../res/sensor_fusion/calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time_data, distance, velocity_command, \
        raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

    sonar1_sen = Sonar1Sensor(
        distance, sonar1, use_saved=False, should_plot=True)
    print("Error Variance: ", sonar1_sen.model_error_var)

    plt.show()