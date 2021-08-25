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

from re import X
import matplotlib.pyplot as plt
from numpy.lib.function_base import msort
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.formula.api import ols

#*****************************************************************************
#
# The following is initialisation
#
#*****************************************************************************

# Load data sonar1
filename = 'part_a/sonar1-calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
distance, sonar1 = data.T

#*****************************************************************************
#
# The following is for functions
#
#*****************************************************************************

# Parameter Function for least square function
def fun_sonar1(k, x, y):
    return (k[0] * x) + k[1] - y

# Linear Sensor Model
def generate_data_sonar1(x, k1, k2, noise=0, n_outliers=0, random_state=0):
    y = (k1 * x) + k2
    return y

def cal_mse(actual_data, model_data):
    mse = mean_squared_error(actual_data, model_data)
    return mse

def error_array(actual_data, model_data):
     err_array = (actual_data - model_data)
     return err_array

def detect_outlier(data):
    # find q1 and q3 values
    q1, q3 = np.percentile(sorted(data), [25, 75])
 
    # compute IRQ
    iqr = q3 - q1
 
    # find lower and upper bounds
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
 
    outliers = [x for x in data if x <= lower_bound or x >= upper_bound]
 
    return outliers
    
#*****************************************************************************
#
# The following is for main program
#
#*****************************************************************************

# Initial Guess
x0 = [1, 1, 1]

# Perform Least Square Regress
res_robust_sonar1 = least_squares(fun_sonar1, x0, loss='soft_l1', f_scale=0.1, args=(distance, sonar1))
y_n_sonar1 = generate_data_sonar1(distance, *res_robust_sonar1.x)

# MSE & Variance
err_mse = cal_mse(sonar1, y_n_sonar1)
err_arr = error_array(sonar1, y_n_sonar1)
err_var = err_mse/len(sonar1)

# Outliers
outliers_x = []
outliers_y = []
for i in range(0, len(distance)):
    if(abs(sonar1[i] - y_n_sonar1[i]) > 1):
        outliers_x.append(distance[i])
        outliers_y.append(sonar1[i])

#*****************************************************************************
#
# The following is for graph plotting
#
#*****************************************************************************

# Plot data
fig1 = plt.figure(1)
plt.plot(distance, sonar1, '.', markersize=1, color="black")
plt.plot(distance, y_n_sonar1, '.', color="red")
plt.plot(outliers_x, outliers_y, 'X', color="green")
plt.title('Linear Sensor Calibration Data') 
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