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
    def __init__(self, distance, measurement, polynomial_order=2, segment_point1=0.21, segment_point2=0.54):
        self.distance = distance
        self.measurement = measurement
        self.brkpt_dist1 = segment_point1
        self.brkpt_dist2 = segment_point2
        self.poly_order = polynomial_order

        self.brkpt1 = find_nearest_index(self.distance, self.brkpt_dist1)
        self.brkpt2 = find_nearest_index(self.distance, self.brkpt_dist2)

        self.lin_ransac1 = make_pipeline(PolynomialFeatures(self.poly_order), RANSACRegressor(random_state=1))
        self.lin_ransac2 = make_pipeline(PolynomialFeatures(self.poly_order), RANSACRegressor(random_state=1))
        self.poly_ransac = make_pipeline(PolynomialFeatures(self.poly_order), RANSACRegressor(random_state=1))

        self.plots_init()
        self.calculate()
        self.plots_draw()
    
    def segment_point_update(self, val):
        self.brkpt1 = find_nearest_index(self.distance, self.brkpt_slider1.val)
        self.brkpt2 = find_nearest_index(self.distance, self.brkpt_slider2.val)
        self.calculate()
        self.plots_draw()

    def calculate(self):
        self.lin_dist1 = self.distance[0:self.brkpt1+100]
        self.lin_meas1 = self.measurement[0:self.brkpt1+100]

        self.lin_dist2 = self.distance[self.brkpt1-100:self.brkpt2+100]
        self.lin_meas2 = self.measurement[self.brkpt1-100:self.brkpt2+100]

        self.poly_dist = self.distance
        self.poly_meas = self.measurement

        self.lin_ransac1.fit(self.lin_dist1.reshape(-1,1), self.lin_meas1)
        self.lin_pred1 = self.lin_ransac1.predict(self.lin_dist1.reshape(-1,1))
        self.lin_inlier_mask1 = self.lin_ransac1.steps[1][1].inlier_mask_
        self.lin_outlier_mask1 = np.logical_not(self.lin_inlier_mask1)

        self.lin_ransac2.fit(self.lin_dist2.reshape(-1,1), self.lin_meas2)
        self.lin_pred2 = self.lin_ransac2.predict(self.lin_dist2.reshape(-1,1))
        self.lin_inlier_mask2 = self.lin_ransac2.steps[1][1].inlier_mask_
        self.lin_outlier_mask2 = np.logical_not(self.lin_inlier_mask2)

        self.poly_ransac.fit(self.poly_dist.reshape(-1,1), self.poly_meas)
        self.poly_pred = self.poly_ransac.predict(self.poly_dist.reshape(-1,1))
        self.poly_inlier_mask = self.poly_ransac.steps[1][1].inlier_mask_
        self.poly_outlier_mask = np.logical_not(self.poly_inlier_mask)

        self.lin_dist_inlier1 = np.delete(self.lin_dist1, self.lin_outlier_mask1)
        self.lin_meas_inlier1 = np.delete(self.lin_meas1, self.lin_outlier_mask1)

        self.lin_dist_inlier2 = np.delete(self.lin_dist2, self.lin_outlier_mask2)
        self.lin_meas_inlier2 = np.delete(self.lin_meas2, self.lin_outlier_mask2)

        self.poly_dist_inlier = np.delete(self.poly_dist, self.poly_outlier_mask)
        self.poly_meas_inlier = np.delete(self.poly_meas, self.poly_outlier_mask)

        self.dist_inlier = np.concatenate((self.lin_dist_inlier1, self.lin_dist_inlier2, self.poly_dist_inlier))
        self.meas_inlier = np.concatenate((self.lin_meas_inlier1, self.lin_meas_inlier2, self.poly_meas_inlier))

        self.pred_inlier = np.piecewise(self.dist_inlier, [self.dist_inlier <= self.brkpt_dist1, (self.brkpt_dist1 < self.dist_inlier) & (self.dist_inlier < self.brkpt_dist2), self.dist_inlier >= self.brkpt_dist2],
                                        [lambda dist_inlier: self.lin_ransac1.predict(dist_inlier.reshape(-1, 1)),
                                        lambda dist_inlier: self.lin_ransac2.predict(dist_inlier.reshape(-1, 1)),
                                         lambda dist_inlier: self.poly_ransac.predict(dist_inlier.reshape(-1, 1))])

        self.error_inlier = self.meas_inlier - self.pred_inlier
        self.err_variance = np.var(self.error_inlier)
        print(self.err_variance)

    def plots_init(self):
        self.inliers_fig, self.inliers_ax = plt.subplots()
        plt.title("Inliers and Outliers of IR3")

        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.ir3_fig, self.ir3_ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        plt.title("IR3")
        self.ir3_ax_brkpt1 = plt.axes([0.25, 0.25, 0.65, 0.03])
        self.ir3_ax_brkpt2 = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.brkpt_slider1 = Slider(self.ir3_ax_brkpt1, 'Breakpoint 1', 0.0, 1.0, 0.2)
        self.brkpt_slider2 = Slider(self.ir3_ax_brkpt2, 'Breakpoint 2', 0.0, 1.0, 0.4)
        self.brkpt_slider1.on_changed(self.segment_point_update)
        self.brkpt_slider2.on_changed(self.segment_point_update)

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
        self.ir3_ax.plot(self.distance, self.measurement, '.')
        self.ir3_ax.plot(self.lin_dist1, self.lin_pred1, color='orange')
        self.ir3_ax.plot(self.lin_dist2, self.lin_pred2, color='orange')
        self.ir3_ax.plot(self.poly_dist, self.poly_pred, color='red')
        self.ir3_ax.plot(self.dist_inlier, self.pred_inlier, color='gold')
        self.ir3_fig.canvas.draw()


ir3_sen = IR3SensorModel(distance, raw_ir3)
plt.show()
