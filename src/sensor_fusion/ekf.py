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
# Module Description:
# !Make sure to run all of the sensor scripts before running this script
# !as the parameters are only loaded from 'data/*' files in this module
# The parameters such as sensor model is calculated before by running each
# sensor model python script. The model and parameters only loaded in this
# module
################################################################################
import numpy as np
import matplotlib.pyplot as plt

from motion_model import MotionModel
from sonar1 import Sonar1Sensor
from ir3 import Ir3Sensor
from ir4 import Ir4Sensor

# Load data
filename = '../../res/sensor_fusion/training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

'''
filename = '../../res/sensor_fusion/training1.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
     sonar1, sonar2 = data.T
'''

step_num = np.size(index)

sonar1_sen = Sonar1Sensor()
ir3_sen = Ir3Sensor()
ir4_sen = Ir4Sensor()
motion_model = MotionModel()

#motion_model_var = motion_model.get_variance()
#motion_model_var = 1.8 * 10 ** -7
#motion_model_var = 2.4 * 10 ** -6
#motion_model_var = 6.1 * 10 ** -6
motion_model_var = 6 * 10 ** -6

mean_est = np.zeros(step_num)
plot_k = np.zeros(step_num)
plot_w1 = np.zeros(step_num)
plot_w2 = np.zeros(step_num)
plot_w3 = np.zeros(step_num)

plot_meas_var = np.zeros(step_num)
plot_motion_var = np.zeros(step_num)

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

    a = sonar1_sen.var_estimator(mean_x_prior)
    b = ir3_sen.var_estimator(mean_x_prior)
    c = ir4_sen.var_estimator(mean_x_prior)

    w1 = 1 / sonar1_sen.var_estimator(mean_x_prior) / (1 / sonar1_sen.var_estimator(mean_x_prior) + 1 / ir3_sen.var_estimator(mean_x_prior) + 1 / ir4_sen.var_estimator(mean_x_prior))
    w2 = 1 / ir3_sen.var_estimator(mean_x_prior) / (1 / sonar1_sen.var_estimator(mean_x_prior) + 1 / ir3_sen.var_estimator(mean_x_prior) + 1 / ir4_sen.var_estimator(mean_x_prior))
    w3 = 1 / ir4_sen.var_estimator(mean_x_prior) / (1 / sonar1_sen.var_estimator(mean_x_prior) + 1 / ir3_sen.var_estimator(mean_x_prior) + 1 / ir4_sen.var_estimator(mean_x_prior))

    plot_w1[n] = w1
    plot_w2[n] = w2
    plot_w3[n] = w3

    mean_x_meas = w1 * sonar1_est + w2 * ir3_est + w3 * ir4_est
    var_x_meas = 1 / (1 / sonar1_sen.var_estimator(mean_x_prior) + 1 / ir3_sen.var_estimator(mean_x_prior) + 1 / ir4_sen.var_estimator(mean_x_prior) )

    plot_meas_var[n] = var_x_meas
    plot_motion_var[n] = var_x_prior

    K = ( 1 / var_x_meas ) / (1 / var_x_meas + 1 / var_x_prior)

    mean_x_posterior = K * mean_x_meas + (1 - K ) * mean_x_prior
    var_x_posterior = 1 / (1 / var_x_meas + 1 / var_x_prior)

    if(mean_x_posterior < 0.1):
        mean_x_posterior = 0.1
    if(mean_x_posterior > 3):
        mean_x_posterior = 3

    mean_est[n] = mean_x_posterior
    plot_k[n] = K

fig1, ax1 = plt.subplots()
plt.plot(distance, label='True Distance')
#plt.plot(sonar2)
#plt.plot(sonar1)
plt.plot(mean_est, label='Estimated Distance')
plt.plot(plot_k, label='Kalman Gain')
plt.legend(loc='upper right')

fig2, ax2 = plt.subplots()
plt.plot(plot_k, label='Kalman Gain')
plt.legend(loc='upper right')

fig3, ax3 = plt.subplots()
plt.plot(plot_w1, label='Sonar Sensor Weight')
plt.plot(plot_w2 , label='IR3 Sensor Weight')
plt.plot(plot_w3 , label='IR4 Sensor Weight')
plt.legend(loc='upper right')

fig4, ax4 = plt.subplots()
plt.plot(plot_meas_var[10:-1], label='Measurement Variance')
plt.plot(plot_motion_var[10:-1] , label='Motion Model Variance')
plt.legend(loc='upper right')

plt.show()