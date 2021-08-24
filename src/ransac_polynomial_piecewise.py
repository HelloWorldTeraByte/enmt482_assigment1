import numpy as np
import matplotlib.pyplot as plt
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

brkpt = find_nearest_index(distance, 0.4)

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
pol_inlier_mask = poly_ransac.steps[1][1].inlier_mask_
pol_outlier_mask = np.logical_not(pol_inlier_mask)

poly_dist_inlier = np.delete(poly_dist, pol_outlier_mask)
poly_meas_inlier = np.delete(poly_meas, pol_outlier_mask)
poly_pred_inlier = poly_ransac.predict(poly_dist_inlier.reshape(-1,1))
poly_error_inlier = poly_meas_inlier - poly_pred_inlier

print("Distance", np.size(distance))
print("Distance inlier size", np.size(poly_dist_inlier))
print("Preditction inlier size", np.size(poly_pred_inlier))
print("Error size", np.size(poly_error_inlier))

fig, axes = plt.subplots(2, 3)
fig.suptitle('Calibration data')

axes[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
axes[0, 0].set_title('IR1')

axes[0, 1].plot(distance, raw_ir2, '.', alpha=0.2)
# axes[0, 1].plot(lin_dist, lin_pred)
# axes[0, 1].plot(poly_dist, poly_pred)
# axes[0, 1].scatter(pol_out_x_data, pol_out_y_data, color='red', marker='.')
axes[0, 1].set_title('IR2')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].plot(lin_dist, lin_pred, color='orange')
axes[0, 2].plot(poly_dist, poly_pred, color='red')
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')

poly_dist1 = np.delete(poly_dist, pol_outlier_mask)
poly_meas1 = np.delete(poly_meas, pol_outlier_mask)

poly_dist2 = np.delete(poly_dist, pol_inlier_mask)
poly_meas2 = np.delete(poly_meas, pol_inlier_mask)

fig2, axes2 = plt.subplots(2)
axes2[0].scatter(poly_dist_inlier, poly_error_inlier)
axes2[1].hist(poly_error_inlier, 100)

plt.figure()
plt.scatter(poly_dist1, poly_meas1)
plt.scatter(poly_dist2, poly_meas2)

plt.show()