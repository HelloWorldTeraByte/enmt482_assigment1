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

# Load data
filename = 'part_a/sonar1-calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
#index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
#    sonar1, sonar2 = data.T
distance, sonar1 = data.T


def fun(k1, k2, x):
    return k1/(k2+x)


x0 = np.ones(1)
res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(distance, sonar1))

y = generate_data(*res_robust.k1, *res_robust.k2, *res_robust.x)
plt.plot(distance, y, color="red")
plt.ylabel('da')
plt.show()
print(y)

def generate_data(k1, k2, x, noise=0, n_outliers=0, random_state=0):
    y = k1/(k2+x)
    #rnd = np.random.RandomState(random_state)
    #error = noise * rnd.randn(t.size)
    #outliers = rnd.randint(0, t.size, n_outliers)
    #error[outliers] *= 35
    return y
