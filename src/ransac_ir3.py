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
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

brkpt_dist = 0.4
brkpt = find_nearest_index(distance, brkpt_dist)

lin_dist = distance[0:brkpt]
lin_meas = raw_ir3[0:brkpt]

poly_dist = distance
poly_meas = raw_ir3

lin_ransac = RANSACRegressor(random_state=1)
lin_ransac.fit(lin_dist.reshape(-1,1), lin_meas)
lin_pred = lin_ransac.predict(lin_dist.reshape(-1,1))
lin_inlier_mask = lin_ransac.inlier_mask_
lin_outlier_mask = np.logical_not(lin_inlier_mask)

poly_ransac = make_pipeline(PolynomialFeatures(2), RANSACRegressor(random_state=1))
poly_ransac.fit(poly_dist.reshape(-1,1), poly_meas)
#mse = mean_squared_error(ransac.predict(distance.reshape(-1,1)), raw_ir3)
poly_pred = poly_ransac.predict(poly_dist.reshape(-1,1))
poly_inlier_mask = poly_ransac.steps[1][1].inlier_mask_
poly_outlier_mask = np.logical_not(poly_inlier_mask)

lin_dist_inlier = np.delete(lin_dist, lin_outlier_mask)
lin_meas_inlier = np.delete(lin_meas, lin_outlier_mask)

poly_dist_inlier = np.delete(poly_dist, poly_outlier_mask)
poly_meas_inlier = np.delete(poly_meas, poly_outlier_mask)

dist_inlier = (np.concatenate((lin_dist_inlier, poly_dist_inlier)))
meas_inlier = (np.concatenate((lin_meas_inlier, poly_meas_inlier)))

pred_inlier = np.piecewise(dist_inlier, [dist_inlier < 0.4, dist_inlier >= 0.4], \
    [lambda dist_inlier: lin_ransac.predict(dist_inlier.reshape(-1,1)), \
        lambda dist_inlier: poly_ransac.predict(dist_inlier.reshape(-1,1))])

error_inlier = meas_inlier - pred_inlier

plt.figure()
plt.title("Inliers of Ir3")
plt.scatter(distance, raw_ir3, linewidths=0.1)
plt.scatter(dist_inlier, meas_inlier, linewidths=0.1)

fig_err, ax_err = plt.subplots(2)
fig_err.suptitle("Errors")
ax_err[0].scatter(dist_inlier, error_inlier)
ax_err[1].hist(error_inlier, 100)


def brkpt_update(val):
    pass


fig_ir3, ax_ir3 = plt.subplots()
plt.subplots_adjust(bottom=0.35)

plt.plot(distance, raw_ir3, '.', alpha=0.2)
plt.plot(lin_dist, lin_pred, color='orange')
plt.plot(poly_dist, poly_pred, color='red')

ax_ir3_brkpt = plt.axes([0.25, 0.2, 0.65, 0.03])
brkpt_slider = Slider(ax_ir3_brkpt, 'Breakpoint', 0.0, 1.0, 0.6)
brkpt_slider.on_changed(brkpt_update)
plt.title("IR3")
fig, axes = plt.subplots(2, 3)
fig.suptitle('Calibration data')

axes[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
axes[0, 0].set_title('IR1')

axes[0, 1].plot(distance, raw_ir2, '.', alpha=0.2)
axes[0, 1].set_title('IR2')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')

plt.show()