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
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data ir3
filename = 'part_a/ir3-calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
distance, raw_ir3 = data.T


#*****************************************************************************
#
# The following is for functions
#
#*****************************************************************************

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def fun_lin_ir3(x, t, y):
    return x[0]*t + x[1] - y

def fun_pol_ir3(x, t, y):
    return x[0] + (x[1]*t) + (x[2]*t**2) + (x[3]*t**3) - y

def generate_data_ir3_linear(x, k1, k2, noise=0, n_outliers=0, random_state=0):
    y = (k1 * x) + k2
    #rnd = np.random.RandomState(random_state)
    #error = noise * rnd.randn(t.size)
    #outliers = rnd.randint(0, t.size, n_outliers)
    #error[outliers] *= 35
    return y

def generate_data_ir3_poly(x, k1, k2, k3, k4, noise=0, n_outliers=0, random_state=0):
    #y = k1/(x + k2)
    y = k1 + (k2*x) + (k3*x**2) + (k4*x**3)
    #rnd = np.random.RandomState(random_state)
    #error = noise * rnd.randn(t.size)
    #outliers = rnd.randint(0, t.size, n_outliers)
    #error[outliers] *= 35
    return y

def get_err_array(model_data, actual_data):
    err_arr = (actual_data - model_data)
    return err_arr

def cal_err(dataset_opt, dataset_actual, no_of_data):
    i = 0
    error_ir3_total = 0
    for i in range(0, no_of_data):
        error_ir3_total += (dataset_actual[i] - dataset_opt[i])
    return error_ir3_total

def cal_err_piecewise_mean(err_tot_lin, data_count_lin, err_tot_pol, data_count_pol):
    err_tot = err_tot_lin + err_tot_pol
    data_count_tot = data_count_lin + data_count_pol
    error_piecewise_mean = err_tot/data_count_tot
    return error_piecewise_mean

def cal_err_sqr(dataset_opt, dataset_actual, no_of_data):
    i = 0
    error_ir3_sqr_total = 0
    for i in range(0, no_of_data):
        error_ir3_sqr_total += (dataset_opt[i] - dataset_actual[i])**2
    return error_ir3_sqr_total

def cal_err_piecewise_var(err_tot_lin, data_count_lin, err_tot_pol, data_count_pol):
    err_tot = err_tot_lin + err_tot_pol
    data_count_tot = data_count_lin + data_count_pol
    error_piecewise_mean = err_tot/data_count_tot
    return error_piecewise_mean

#*****************************************************************************
#
# The following is for main code
#
#*****************************************************************************

# Split Data into linear and Polynomial sides
brkpt = find_nearest_index(distance, 0.2)
model_lin_x = distance[0:brkpt]
model_lin_y = raw_ir3[0:brkpt]

model_pol_x = distance[brkpt:]
model_pol_y = raw_ir3[brkpt:]

# Performing Robust LSR to linear side
x0_lin = [0, 0, 0]
res_lin_ir3 = least_squares(fun_lin_ir3, x0_lin, loss='soft_l1', f_scale=0.1, args=(model_lin_x, model_lin_y))
y_lin_ir3 = generate_data_ir3_linear(model_lin_x, *res_lin_ir3.x)

# Performing Robust LSR to Polynomial side
x0_pol = [0, 0, 0, 0]
res_poly_ir3 = least_squares(fun_pol_ir3, x0_pol, loss='soft_l1', f_scale=0.1, args=(model_pol_x, model_pol_y))
y_pol_ir3 = generate_data_ir3_poly(model_pol_x, *res_poly_ir3.x)

# Combining the linear and nonlinear data
err_lin_tot = cal_err(y_lin_ir3, model_lin_y, len(model_lin_x))
err_pol_tot = cal_err(y_pol_ir3, model_pol_y, len(model_pol_x))
err_mean = cal_err_piecewise_mean(err_lin_tot, len(model_lin_x), err_pol_tot, len(model_pol_x))
err_sqr_lin_tot = cal_err_sqr(y_lin_ir3, model_lin_y, len(model_lin_x))
err_sqr_pol_tot = cal_err_sqr(y_pol_ir3, model_pol_y, len(model_pol_x))
err_var = cal_err_piecewise_var(err_sqr_lin_tot, len(model_lin_x), err_sqr_pol_tot, len(model_pol_x))

# Get error array
err_arr = []
lin_err_arr = get_err_array(y_lin_ir3, model_lin_y)
pol_err_arr = get_err_array(y_pol_ir3, model_pol_y)
err_arr = [*lin_err_arr, *pol_err_arr]

#*****************************************************************************
#
# The following is for graph plotting
#
#*****************************************************************************

fig1 = plt.figure(1)
plt.plot(model_lin_x, y_lin_ir3, '.', markersize=1, color="red")
plt.plot(model_pol_x, y_pol_ir3, '.', markersize=1, color="red")
plt.plot(distance, raw_ir3, '.', markersize=1, color="black")
plt.title('Non-Linear Sensor Calibration Data') 
plt.ylabel('Sensor Data')
plt.xlabel('Distance (m)')

# Plot Error
fig2 = plt.figure(2)
plt.hist(err_arr, bins=50, density=True)
plt.title('Histogram of Errors between Measured Data and Model') 
plt.ylabel('Count')
plt.xlabel('Measurement Error')

fig3 = plt.figure(3)
plt.plot(distance, err_arr, '.')
plt.title('Errors between Measured Data and Model') 
plt.ylabel('Measurement Error')
plt.xlabel('distance')

plt.show()

