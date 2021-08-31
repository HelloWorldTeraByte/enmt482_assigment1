import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splev, splrep
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_squared_error

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
        self.ransac = RANSACRegressor(Ir4Regressor(), random_state=0, min_samples=1000)

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


    ir4_sen = Ir4Sensor(distance, raw_ir4, should_plot=True)
    print("Error Variance: ", ir4_sen.error_var)
    plt.show()