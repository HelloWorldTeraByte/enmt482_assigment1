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

# Load data
filename = '/home/helloworldterabyte/projects/enmt482-2021_robotic_assignment/data/calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

step_num = 2000
dt = 0.1

motion_model_var = 2 * 10 ** -5

sonar1_sen = Sonar1Sensor(distance, sonar1)
ir3_sen = Ir3Sensor(distance, raw_ir3)
ir4_sen = Ir4Sensor(distance, raw_ir4)

mean_x_posterior = 0.1
var_x_posterior = 10 ** 2

mean_est = np.zeros(step_num)
plot_k = np.zeros(step_num)

for n in range(step_num):
    # Predict
    mean_x_prior = mean_x_posterior + velocity_command[n] * dt
    var_x_prior = var_x_posterior + motion_model_var

    # Update
    sonar1_est = sonar1_sen.x_est_mle(sonar1[n])
    ir3_est = ir3_sen.x_est_mle(raw_ir3[n], mean_x_posterior)
    ir4_est = ir4_sen.x_est_mle(raw_ir4[n], mean_x_posterior)

    w1 = 1 / sonar1_sen.error_var / (1 / sonar1_sen.error_var + 1 / ir3_sen.error_var + 1 / ir4_sen.error_var)
    w2 = 1 / ir3_sen.error_var / (1 / sonar1_sen.error_var + 1 / ir3_sen.error_var + 1 / ir4_sen.error_var)
    w3 = 1 / ir4_sen.error_var / (1 / sonar1_sen.error_var + 1 / ir3_sen.error_var + 1 / ir4_sen.error_var)

    mean_x_meas = w1 * sonar1_est + w2 * ir3_est + w3 * ir4_est
    var_x_meas = 1 / ( 1 / sonar1_sen.var_estimator() + 1 / ir3_sen.var_estimator_at_x0(mean_x_posterior) + 1 / ir4_sen.var_estimator_at_x0(mean_x_posterior) )

    '''
    K = ( 1 / var_x_meas ) / (1 / var_x_meas + 1 / var_x_prior)

    mean_x_posterior = K * mean_x_meas + (1 - K ) * mean_x_prior
    var_x_posterior =  1  / (1 / var_x_meas + 1 / var_x_prior)
    '''

    K = var_x_prior / ( var_x_meas + var_x_prior )
    mean_x_posterior = mean_x_prior + K * ( mean_x_meas - mean_x_prior)
    var_x_posterior = (1 - K) * var_x_prior

    mean_est[n] = mean_x_posterior
    plot_k[n] = K

fig1, ax1 = plt.subplots()
plt.plot(distance)
plt.plot(mean_est)
plt.plot(plot_k)
plt.show()