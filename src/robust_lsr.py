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
from scipy.optimize import least_squares
import numpy as np

# Load data sonar1
filename = 'part_a/sonar1-calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
#index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
#    sonar1, sonar2 = data.T
distance, sonar1 = data.T

# Load data ir3
filename = 'part_a/ir3-calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
distance, ir3 = data.T

def fun_sonar1(x, t, y):
    return (x[0] * t) + x[1] - y

def generate_data_sonar1(x, k1, k2, noise=0, n_outliers=0, random_state=0):
    y = (k1 * x) + k2
    #rnd = np.random.RandomState(random_state)
    #error = noise * rnd.randn(t.size)
    #outliers = rnd.randint(0, t.size, n_outliers)
    #error[outliers] *= 35
    return y


x0 = [1, 1, 1]

res_robust_sonar1 = least_squares(fun_sonar1, x0, loss='soft_l1', f_scale=0.1, args=(distance, sonar1))

if (res_robust_sonar1.success):
    print("Success")
    print("x = ", res_robust_sonar1.x) 
    print("y = ", res_robust_sonar1.fun)

y_n_sonar1 = generate_data_sonar1(distance, *res_robust_sonar1.x)
#plt.plot(distance, sonar1, '.', markersize=1, color="black")
#plt.plot(distance, y_n_sonar1, color="red")


def fun_ir3(x, t, y):
    return x[0]/(t + x[1]) - y

def generate_data_ir3(x, k1, k2, noise=0, n_outliers=0, random_state=0):
    y = k1/(x + k2)
    #rnd = np.random.RandomState(random_state)
    #error = noise * rnd.randn(t.size)
    #outliers = rnd.randint(0, t.size, n_outliers)
    #error[outliers] *= 35
    return y

res_robust_ir3 = least_squares(fun_ir3, x0, loss='soft_l1', f_scale=1, args=(distance, ir3))
y_n_ir3 = generate_data_ir3(distance, *res_robust_ir3.x)
plt.plot(distance, ir3, '.', markersize=1, color="black")
plt.plot(distance, y_n_ir3, color="red")
plt.ylabel('da')
plt.show()