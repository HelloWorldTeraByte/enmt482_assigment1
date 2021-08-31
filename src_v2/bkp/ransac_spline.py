import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import RANSACRegressor
from scipy.stats import median_abs_deviation

from spline_sensor_regressor import SplineSensorRegressor

class Ir4Sensor(object):
    def __init__(self, distance, measurement, should_plot = False, smooth=130):
        self.distance = distance
        self.measurement = measurement
        self.smooth_val = smooth
        self.ransac = RANSACRegressor(SplineSensorRegressor(smooth=self.smooth_val), random_state=0, min_samples=10,residual_threshold=(median_abs_deviation(self.measurement)+0.6))

        self.ransac.fit(self.distance.reshape(-1,1), self.measurement)
        self.inlier_mask = self.ransac.inlier_mask_
        self.outlier_mask = np.logical_not(self.inlier_mask)

        self.meas_inliers = np.delete(self.measurement, self.outlier_mask)
        self.dist_inliers = np.delete(self.distance, self.outlier_mask)

        self.ransac_pred = self.ransac.predict(self.dist_inliers.reshape(-1,1))
        self.errors = self.meas_inliers - self.ransac_pred
        self.error_var = np.var(self.errors)

        if(should_plot):
            self.plots_init()
            self.plots_draw()
    
    def plots_init(self):
        self.liers_fig, self.liers_ax = plt.subplots()
        plt.title("Inliers and Outliers")

        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.sensor_fig, self.sensor_ax = plt.subplots()
        plt.title("IR4")

    def plots_draw(self):
        self.liers_ax.scatter(self.distance, self.measurement)
        self.liers_ax.scatter(self.dist_inliers, self.meas_inliers)

        self.err_ax[0].scatter(self.dist_inliers, self.errors)
        self.err_ax[1].hist(self.errors, 100)

        self.sensor_ax.plot(self.distance, self.measurement, '.')
        self.sensor_ax.plot(self.dist_inliers, self.ransac_pred, color='red', linewidth=2)


if __name__ == "__main__":
    filename = '/home/helloworldterabyte/projects/enmt482-2021_robotic_assignment/part_a/calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time_data, distance, velocity_command, \
        raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T


    ir4_sen = Ir4Sensor(distance, raw_ir4, should_plot=True, smooth=130)
    plt.show()