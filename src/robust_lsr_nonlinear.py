#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~
#************************************************************************/
#*                                                                      */
#                            sensor_data_plotter.py
#*                                                                      */
#************************************************************************/


#   Authors:        Jason Ui
#                   Randipa
#
#
#
#   Date created:       16/08/2021
#   Date Last Modified: 16/08/2021


#************************************************************************/

#  Module Description:
#
#

import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
import numpy as np

#*****************************************************************************
#
# The following is initialisation
#
#*****************************************************************************

# Load data sonar1
filename = 'part_a/ir3-calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
distance, ir3 = data.T

#*****************************************************************************
#
# The following is for functions
#
#*****************************************************************************

# Parameter Function for least square function
def fun_ir3(x, t, y):
    return x[0]/(x[1] + t) - y

# Non-Linear Sensor Model
def generate_data_ir3(x, k1, k2, noise=0, n_outliers=0, random_state=0):
    y = k1/(k2 + x)
    #rnd = np.random.RandomState(random_state)
    #error = noise * rnd.randn(t.size)
    #outliers = rnd.randint(0, t.size, n_outliers)
    #error[outliers] *= 35
    return y

def cal_mse(actual_data, model_data):
    mse = mean_squared_error(actual_data, model_data)
    return mse

def error_array(actual_data, model_data):
    err_arr = []
    err_arr = (actual_data - model_data)
    return err_arr

#*****************************************************************************
#
# The following is for main program
#
#*****************************************************************************

# Initial Guess
x0 = [1, 1, 1]

# Perform Least Square Regress
res_robust_ir3 = least_squares(fun_ir3, x0, loss='soft_l1', f_scale=0.1, args=(distance, ir3))
y_n_ir3 = generate_data_ir3(distance, *res_robust_ir3.x)

# MSE & Variance
err_mse = cal_mse(ir3, y_n_ir3)
err_arr = error_array(ir3, y_n_ir3)
err_var = err_mse/len(ir3)

#*****************************************************************************
#
# The following is for graph plotting
#
#*****************************************************************************

# Plot data
fig1 = plt.figure(1)
plt.plot(distance, ir3, '.', markersize=1, color="black")
plt.plot(distance, y_n_ir3, color="red")
plt.title('Non-Linear Sensor Calibration Data') 
plt.ylabel('Sensor Data')
plt.xlabel('Distance (m)')

# Plot error
fig2 = plt.figure(2)
plt.hist(err_arr, bins=200, density=True)
plt.title('Histogram of Errors between Measured Data and Model') 
plt.ylabel('Count')
plt.xlabel('Measurement Error')

fig3 = plt.figure(3)
plt.plot(distance, err_arr, '.')
plt.title('Errors between Measured Data and Model') 
plt.ylabel('Measurement Error')
plt.xlabel('distance')

plt.show()