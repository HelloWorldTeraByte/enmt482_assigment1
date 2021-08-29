#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~
#************************************************************************/
#*                                                                      */
#                            extended_kalman_filter.py
#*                                                                      */
#************************************************************************/


#   Authors:        Jason Ui
#                   Randipa
#
#
#
#   Date created:       27/08/2021
#   Date Last Modified: 27/08/2021


#************************************************************************/

#  Module Description:
#
#

import statistics
import numpy as np
from matplotlib.pyplot import subplots, show, savefig
import matplotlib.pyplot as plt
from src.motion_model import MotionModel
from src.ransac_ir3_spline import IR3SensorModel

#*****************************************************************************
#
# The following is to load data
#
#*****************************************************************************

# Calibration
filename_c = 'part_a_old/calibration.csv'
data_c = np.loadtxt(filename_c, delimiter=',', skiprows=1)
index_c, time_c, distance_c, velocity_command_c, raw_ir1_c, raw_ir2_c, raw_ir3_c, raw_ir4_c, \
    sonar1_c, sonar2_c = data_c.T

# Training1
filename_t1 = 'part_a_old/training1.csv'
data_t1 = np.loadtxt(filename_t1, delimiter=',', skiprows=1)
index_t1, time_t1, distance_t1, velocity_command_t1, raw_ir1_t1, raw_ir2_t1, raw_ir3_t1, raw_ir4_t1, \
    sonar1_t1, sonar2_t1 = data_t1.T

# Training2
filename_t2 = 'part_a_old/training2.csv'
data_t2 = np.loadtxt(filename_t2, delimiter=',', skiprows=1)
index_t2, time_t2, distance_t2, velocity_command_t2, raw_ir1_t2, raw_ir2_t2, raw_ir3_t2, raw_ir4_t2, \
    sonar1_t2, sonar2_t2 = data_t2.T


#*****************************************************************************
#
# The following is the function for calculating Kalman Filter Gain
#
#*****************************************************************************

#*****************************************************************************
#
# The following is the function for Kalman Filter's prediction step
# Prediction Step is also known as time update
#
#*****************************************************************************

def prediction(motion_model, prev_belief_post, prev_belief_var_post, process_noise, process_noise_var):

    # Predict the robot's position using the previous estimated position and your motion model
    predicted_belief_prior = prev_belief_post + motion_model + process_noise 

    # Determine the variance of the predicted robot's position
    predicted_belief_var_prior = prev_belief_var_post + process_noise_var
  
    return predicted_belief_prior, predicted_belief_var_prior

#*****************************************************************************
#
# The following is the function for Kalman Filter's update step
# Update step is also known as correction or measurement update
#
#*****************************************************************************
 
def update(sensor_data, curr_belief_prior, prev_belief_var_post, curr_belief_var_prior, process_noise_var, measurement_noise_var):
    
    kf_gain = (prev_belief_var_post + process_noise_var) / (prev_belief_var_post + process_noise_var + measurement_noise_var)

    curr_belief_post =  curr_belief_prior + (kf_gain * sensor_data)

    curr_belief_var_post = (1 - kf_gain) * curr_belief_var_prior

    return curr_belief_post, curr_belief_var_post

#*****************************************************************************
#
# The following is for main program
#
#*****************************************************************************

# Sensor Model & Motion Model
ir3_sen = IR3SensorModel(distance_c, raw_ir3_c)
motion_model_c = MotionModel(distance_t1, time_t1, velocity_command_t1, training_no=0, plot=0)

#motion_model_t1 = MotionModel(distance_t1, time_t1, velocity_command_t1, training_no=1, plot=0)
#motion_model_t2 = MotionModel(distance_t2, time_t2, velocity_command_t2, training_no=2, plot=0)

# Initial Parameters
initial_belief = distance_c[0]
initial_belief_var = np.var(distance_c[0])

belief_array = []
belief_var_array = []
belief_array.append(initial_belief)
belief_var_array.append(initial_belief_var) 
#measurement_covar = 0 #TODO
#process_noise_covar = 0 #TODO

i = 0
for i in range(0, len(distance_c)):
    predicted_belief_prior, predicted_belief_var_prior = prediction(motion_model_c.motion_model, belief_array[i], belief_var_array[i], motion_model_c.process_noise, motion_model_c.process_noise_var)
    curr_belief_post, curr_belief_var_post = update(ir3_sen.meas_inlier[i], predicted_belief_prior, belief_var_array[i], predicted_belief_var_prior, motion_model_c.process_noise_var, ir3_sen.err_variance)
    belief_array.append(curr_belief_post) 
    belief_var_array.append(curr_belief_var_post)

_fig_ = plt.figure(10)
plt.plot(distance_c, velocity_command_c)
plt.plot(distance_c, belief_array)
plt.title('Errors between Measured Data and Model') 
plt.ylabel('Measurement Error')
plt.xlabel('distance')
plt.show(_fig_)
