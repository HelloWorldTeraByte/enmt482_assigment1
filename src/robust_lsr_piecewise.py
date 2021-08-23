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
#return x[0]/(t + x[1]) - y
    return x[0]*t + x[1] - y

def fun_pol_ir3(x, t, y):
#return x[0]/(t + x[1]) - y
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

#*****************************************************************************
#
# The following is for main code
#
#*****************************************************************************

brkpt = find_nearest_index(distance, 0.2)
print(brkpt)

model_lin_x = distance[0:brkpt]
model_lin_y = raw_ir3[0:brkpt]

model_pol_x = distance[brkpt:]
model_pol_y = raw_ir3[brkpt:]

fig1 = plt.figure(1)

x0_lin = [0, 0, 0]
res_lin_ir3 = least_squares(fun_lin_ir3, x0_lin, loss='soft_l1', f_scale=0.1, args=(model_lin_x, model_lin_y))
y_lin_ir3 = generate_data_ir3_linear(model_lin_x, *res_lin_ir3.x)
plt.plot(model_lin_x, y_lin_ir3, '.', markersize=1, color="red")

x0_pol = [0, 0, 0, 0]
res_poly_ir3 = least_squares(fun_pol_ir3, x0_pol, loss='soft_l1', f_scale=0.1, args=(model_pol_x, model_pol_y))
y_poly_ir3 = generate_data_ir3_poly(model_pol_x, *res_poly_ir3.x)
plt.plot(model_pol_x, y_poly_ir3, '.', markersize=1, color="red")

plt.plot(distance, raw_ir3, '.', markersize=1, color="black")
plt.ylabel('da')
plt.show()
