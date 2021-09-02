################################################################################
#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,                 #
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/                  #
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~                #
################################################################################
#                                                                              #
#                               ekf.py                                         # 
#                                                                              #
################################################################################
# Authors:        Jason Ui
#                 Randipa Gunathilake
#
# Date created:       31/08/2021
# Date Last Modified: 31/08/2021
################################################################################
#  Module Description:
#
#
################################################################################
import numpy as np
import matplotlib.pyplot as plt

from sonar1 import Sonar1Sensor
from ir3 import Ir3Sensor
from ir4 import Ir4Sensor

filename_c = '/home/helloworldterabyte/projects/enmt482-2021_robotic_assignment/data/calibration.csv'
#filename_c = '../data/calibration.csv'
data_c = np.loadtxt(filename_c, delimiter=',', skiprows=1)
index_c, time_c, distance_c, velocity_command_c, raw_ir1_c, raw_ir2_c, raw_ir3_c, raw_ir4_c, \
    sonar1_c, sonar2_c = data_c.T

# Load data
filename = '/home/helloworldterabyte/projects/enmt482-2021_robotic_assignment/data/training1.csv'
#filename = '../data/training1.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

step_num = np.size(index)

#motion_model_var = 1.8 * 10 ** -7
motion_model_var = 2.4 * 10 ** -6
#motion_model_var = 6.1 * 10 ** -6

#motion_model_var = 1 * 10 ** -5

sonar1_sen = Sonar1Sensor(distance_c, sonar1_c)
ir3_sen = Ir3Sensor(distance_c, raw_ir3_c)
ir4_sen = Ir4Sensor(distance_c, raw_ir4_c)

mean_est = np.zeros(step_num)
plot_k = np.zeros(step_num)

mean_x_posterior = 0.1
var_x_posterior = 10 ** 2

for n in range(1, step_num):
    # Predict
    mean_x_prior = mean_x_posterior + velocity_command[n] * (time[n] - time[n-1])
    var_x_prior = var_x_posterior + motion_model_var
    #print("Time at n", time[n], "Time at n-1", time[n-1], "dt", time[n] - time[n-1])

    # Update
    sonar1_est = sonar1_sen.x_est_mle(sonar1[n])
    ir3_est = ir3_sen.x_est_mle(raw_ir3[n], mean_x_prior)
    ir4_est = ir4_sen.x_est_mle(raw_ir4[n], mean_x_prior)

    w1 = 1 / sonar1_sen.var_estimator() / (1 / sonar1_sen.var_estimator() + 1 / ir3_sen.var_estimator_at_x0(mean_x_prior) + 1 / ir4_sen.var_estimator_at_x0(mean_x_prior))
    w2 = 1 / ir3_sen.var_estimator_at_x0(mean_x_prior) / (1 / sonar1_sen.var_estimator() + 1 / ir3_sen.var_estimator_at_x0(mean_x_prior) + 1 / ir4_sen.var_estimator_at_x0(mean_x_prior))
    w3 = 1 / ir4_sen.var_estimator_at_x0(mean_x_prior) / (1 / sonar1_sen.var_estimator() + 1 / ir3_sen.var_estimator_at_x0(mean_x_prior) + 1 / ir4_sen.var_estimator_at_x0(mean_x_prior))

    mean_x_meas = w1 * sonar1_est + w2 * ir3_est + w3 * ir4_est
    var_x_meas = 1 / (1 / sonar1_sen.var_estimator() + 1 / ir3_sen.var_estimator_at_x0(mean_x_prior) + 1 / ir4_sen.var_estimator_at_x0(mean_x_prior) )

    K = ( 1 / var_x_meas ) / (1 / var_x_meas + 1 / var_x_prior)

    mean_x_posterior = K * mean_x_meas + (1 - K ) * mean_x_prior
    var_x_posterior = 1 / (1 / var_x_meas + 1 / var_x_prior)

    if(mean_x_posterior < 0.1):
        mean_x_posterior = 0.1
    if(mean_x_posterior > 3):
        mean_x_posterior = 3


    '''
    K = var_x_prior / ( var_x_meas + var_x_prior )
    mean_x_posterior = mean_x_prior + K * ( mean_x_meas - mean_x_prior)
    var_x_posterior = (1 - K) * var_x_prior
    '''

    mean_est[n] = mean_x_posterior
    plot_k[n] = K

fig1, ax1 = plt.subplots()
plt.plot(distance)
plt.plot(mean_est)
plt.plot(plot_k)

fig2, ax2 = plt.subplots()
plt.plot(plot_k)
print(np.average(plot_k))

plt.show()
