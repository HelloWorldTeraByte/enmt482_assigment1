import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time_data, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class IR3SensorModel(object):
    def __init__(self, distance, measurement, polynomial_order=2, segment_point=0.4):
        self.distance = distance
        self.measurement = measurement
        self.brkpt_dist = segment_point
        self.poly_order = polynomial_order

        self.brkpt = find_nearest_index(self.distance, self.brkpt_dist)

        self.lin_dist = distance[0:self.brkpt]
        self.lin_meas = raw_ir3[0:self.brkpt]

        self.poly_dist = distance
        self.poly_meas = raw_ir3

        self.lin_ransac = RANSACRegressor(random_state=1)
        self.poly_ransac = make_pipeline(PolynomialFeatures(self.poly_order), RANSACRegressor(random_state=1))

        self.plots_init()
        self.calculate()
        self.plots_draw()
    
    def segment_point_update(self, val):
        self.brkpt = find_nearest_index(self.distance, self.brkpt_slider.val)
        self.calculate()
        self.plots_draw()

    def calculate(self):
        self.lin_dist = distance[0:self.brkpt]
        self.lin_meas = raw_ir3[0:self.brkpt]

        self.poly_dist = distance
        self.poly_meas = raw_ir3

        self.lin_ransac.fit(self.lin_dist.reshape(-1,1), self.lin_meas)
        self.lin_pred = self.lin_ransac.predict(self.lin_dist.reshape(-1,1))
        self.lin_inlier_mask = self.lin_ransac.inlier_mask_
        self.lin_outlier_mask = np.logical_not(self.lin_inlier_mask)

        self.poly_ransac.fit(self.poly_dist.reshape(-1,1), self.poly_meas)
        #self.mse = mean_squared_error(ransac.predict(distance.reshape(-1,1)), raw_ir3)
        self.poly_pred = self.poly_ransac.predict(self.poly_dist.reshape(-1,1))
        self.poly_inlier_mask = self.poly_ransac.steps[1][1].inlier_mask_
        self.poly_outlier_mask = np.logical_not(self.poly_inlier_mask)

        self.lin_dist_inlier = np.delete(self.lin_dist, self.lin_outlier_mask)
        self.lin_meas_inlier = np.delete(self.lin_meas, self.lin_outlier_mask)

        self.poly_dist_inlier = np.delete(self.poly_dist, self.poly_outlier_mask)
        self.poly_meas_inlier = np.delete(self.poly_meas, self.poly_outlier_mask)

        self.dist_inlier = np.concatenate((self.lin_dist_inlier, self.poly_dist_inlier))
        self.meas_inlier = np.concatenate((self.lin_meas_inlier, self.poly_meas_inlier))

        self.pred_inlier = np.piecewise(self.dist_inlier, [self.dist_inlier < 0.4, self.dist_inlier >= 0.4],
                                        [lambda dist_inlier: self.lin_ransac.predict(dist_inlier.reshape(-1, 1)),
                                         lambda dist_inlier: self.poly_ransac.predict(dist_inlier.reshape(-1, 1))])

        self.error_inlier = self.meas_inlier - self.pred_inlier

    def plots_init(self):

        self.inliers_fig, self.inliers_ax = plt.subplots()
        plt.title("Inliers and Outliers of IR3")

        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.ir3_fig, self.ir3_ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        plt.title("IR3")
        self.ir3_ax_brkpt = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.brkpt_slider = Slider(self.ir3_ax_brkpt, 'Breakpoint', 0.0, 1.0, 0.4)
        self.brkpt_slider.on_changed(self.segment_point_update)

    def plots_draw(self):
        self.inliers_ax.clear()
        self.inliers_ax.scatter(self.distance, self.measurement, linewidths=0.1)
        self.inliers_ax.scatter(self.dist_inlier, self.meas_inlier, linewidths=0.1)
        self.inliers_fig.canvas.draw()

        self.err_ax[0].clear()
        self.err_ax[1].clear()
        self.err_ax[0].scatter(self.dist_inlier, self.error_inlier)
        self.err_ax[1].hist(self.error_inlier, 100)
        self.err_fig.canvas.draw()

        self.ir3_ax.clear()
        self.ir3_ax.plot(self.distance, self.measurement, '.', alpha=0.2)
        self.ir3_ax.plot(self.lin_dist, self.lin_pred, color='orange')
        self.ir3_ax.plot(self.poly_dist, self.poly_pred, color='red')
        self.ir3_fig.canvas.draw()



ir3_sen = IR3SensorModel(distance, raw_ir3)
plt.show()