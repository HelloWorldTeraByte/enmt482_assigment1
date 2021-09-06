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

from util import find_nearest_index

class Sonar1Sensor(object):
    def __init__(self, distance, measurement, should_plot = False):
        self.distance = distance
        self.measurement = measurement
        self.dist_min = np.min(self.distance)
        self.dist_max = np.max(self.distance)
        self.ransac = RANSACRegressor(random_state=0, residual_threshold=(median_abs_deviation(self.measurement))-0.75)

        self.ransac.fit(self.distance.reshape(-1,1), self.measurement)
        self.inlier_mask = self.ransac.inlier_mask_
        self.outlier_mask = np.logical_not(self.inlier_mask)

        self.meas_inliers = np.delete(self.measurement, self.outlier_mask)
        self.dist_inliers = np.delete(self.distance, self.outlier_mask)

        self.ransac_pred = self.ransac.predict(self.dist_inliers.reshape(-1,1))
        self.errors = self.meas_inliers - self.ransac_pred
        self.error_var = np.var(self.errors)

        # TODO: Testing
        self.ransac_pred2 = self.ransac.predict(self.distance.reshape(-1,1))
        self.errors2 = self.measurement - self.ransac_pred2
        self.error_var2 = np.var(self.errors2)
        self.error_var = self.error_var2

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

        if(should_plot):
            self.plots_init()
            self.plots_draw()

    def x_est_mle(self, z):
        x_est = (z - self.ransac.estimator_.intercept_)/self.ransac.estimator_.coef_[0]
        return x_est

    def var_estimator(self, x):
        #var = self.error_var / (self.ransac.estimator_.coef_[0] ** 2)
        var = splev(x, self.err_spline) / (self.ransac.estimator_.coef_[0] ** 2)
        return var
 
    def plots_init(self):
        self.liers_fig, self.liers_ax = plt.subplots()
        plt.title("Inliers and Outliers")

        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.sensor_fig, self.sensor_ax = plt.subplots()
        plt.title("Sonar1")

        self.bin_err, self.bin_err = plt.subplots()
        plt.title("Bin Errors")

    def plots_draw(self):
        self.liers_ax.scatter(self.distance, self.measurement, s=10)
        self.liers_ax.scatter(self.dist_inliers, self.meas_inliers, s=10)

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

    sonar1_sen = Sonar1Sensor(distance, sonar1, should_plot=True)
    print("Error Variance: ", sonar1_sen.error_var)

    plt.show()