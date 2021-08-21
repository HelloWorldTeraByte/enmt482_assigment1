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

from operator import inv
from matplotlib import pyplot as plt
import numpy as np
import sys
from matplotlib.pyplot import subplots, show
from numpy.core.fromnumeric import transpose 
from random import gauss, seed

#*****************************************************************************
#
# The following is code initialisation
#
#*****************************************************************************

plt.close('all') 
seed(1) #For random number

#*****************************************************************************
#
# The following is loading data from csv file(s)
#
#*****************************************************************************

# Load data
filename = '../Assignment 1 Part(A) data and example code/calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

#*****************************************************************************
#
# The following is initialisation of Kalman Filter's variables
#r: only integer scalar arrays can be converted to a scalar index
#*****************************************************************************

# initial estimate
X_0 = 0

# initial error covariance
P_0 = 0

# unknown variables
R = []
Q = []

# Prior Belief/ Prior estimate - current time step
# rough estimate before the measurement update correction. 
X_hat_n_prior = []

# Prior Belief/ Prior estimate - previous time step
X_hat_n_1_prior = []

# Control Signal
U_n = []

# Prior error covariance
# use these prior values in our Measurement Update equations. 
P_n_prior = []

# Prior error covariance - previous time step
P_n_1_prior = []

# Kalman Gain
K_n = 0

P_n = []

#*****************************************************************************
#
# The following is the sensor model
#
#*****************************************************************************

x = np.linspace(0, 10, len(index))

# Signal Value 
H_n = ((3.298 - 0.082)/3.326) * distance + 0.082

#Measurement Noise (random variable)
V_n = gauss(0, 0.1)  #np.random.standard_normal(x.shape)*0.1 #

# Sensor Model
Z_n = H_n + V_n

#*****************************************************************************
#
# The following is the motion model
#
#*****************************************************************************
#TODO
A = 1

#TODO
B = 1

# previous signal
x_n_1 = []

# process noise
wk_1 = []

# motion model for turtlebot
X_n = A * x_n_1 + B * U_n + wk_1

#*****************************************************************************
#
# The following is to plot figure 1 - sensor calibration data
#
#*****************************************************************************

# Format diagrams in presentation into 2 rows and 3 columns
fig1, ax1 = subplots(2, 3)

# Title of the whole diagram presentation
fig1.suptitle('Calibration data')

# Create diagram
ax1[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
ax1[0, 0].set_title('IR1')

ax1[0, 1].plot(distance, raw_ir2, '.', alpha=0.2)
ax1[0, 1].set_title('IR2')

ax1[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
ax1[0, 2].set_title('IR3')

ax1[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
ax1[1, 0].set_title('IR4')

ax1[1, 1].plot(distance, sonar1, '.', alpha=0.2)
ax1[1, 1].set_title('Sonar1')
# Compare sensor data graph with sensor model
ax1[1, 1].plot(distance, H_n, '.',color = "red", alpha=0.1)

ax1[1, 2].plot(distance, sonar2, '.', alpha=0.2)
ax1[1, 2].set_title('Sonar2')

fig1.show()

#*****************************************************************************
#
# The following is to plot Figure 2: Error between measured data and model 
# for sensor calibration data.
#
#*****************************************************************************

# Error
err_sonar1 = Z_n - sonar1

# Format diagrams in presentation into 2 rows and 3 columns
fig2, ax2 = subplots(2, 3)

# Title of the whole diagram presentation
fig2.suptitle('Error between measured data and model')

ax2[0, 0].plot(distance, err_sonar1, '.', alpha=0.2)
ax2[0, 0].set_title('Sonar1 - measured data vs model')
ax2[0, 0].set_ylabel('measurement error u (m)')
ax2[0, 0].set_xlabel('distance x (m)')
#ax2[0, 0].xlim([0, distance])
#ax2[0, 0].ylim([-0.1, 0.1])

fig2.show()

#*****************************************************************************
#
# The following is to plot Figure 3: Random Variable(s) guassian normal
# distribution
#
#*****************************************************************************

# store the random numbers in a list 
gaussian_array = [] 

max_range = 1000
for i in range(max_range): 
    temp = gauss(0, 0.1) #np.random.standard_normal(x.shape) #  gauss(0, 0.1)
    gaussian_array.append(temp)

fig3, ax3 = subplots(2)

ax3[0].plot(gaussian_array, color='black',markersize=2, alpha=0.2)
ax3[0].set_title('Measurement Noise(RV) at each iterations', fontsize=10)
ax3[0].set_ylabel('measurement noise',fontsize=8)
ax3[0].set_xlabel('iterations',fontsize=8)
ax3[0].set_xlim([0, max_range])
ax3[0].set_ylim([-0.25, 0.25])

ax3[1].hist(gaussian_array, bins = 200) 
ax3[1].set_title('Gauss Norm Dist of measurement noise', fontsize=10)
ax3[1].set_ylabel('weighting', fontsize=8)
ax3[1].set_xlabel('measurement noise', fontsize=8)

fig3.show()

#*****************************************************************************
#
# The following is the function for calculating Kalman Filter Gain
#
#*****************************************************************************

def calc_kf_gain(P_n_prior, H, R):
    K_n = (P_n_prior * transpose(H)) * inv(H * P_n_prior * transpose(H) + R) 
    return K_n

#*****************************************************************************
#
# The following is the function for Kalman Filter's prediction step
# Prediction Step is also known as time update
#
#*****************************************************************************

def prediction(X_hat_n_1_prior,P_k_1_prior,A,B,U_n,Q_n):

    # Project the state ahead
    X_hat_n_prior = (A*X_hat_n_1_prior) + (B*U_n)

    # Project the error covariance ahead
    P_n_prior = (A * P_n_1_prior * transpose(A)) + Q_n 
    
    return X_hat_n_prior,P_n_prior

#*****************************************************************************
#
# The following is the function for Kalman Filter's update step
# Update step is also known as correction or measurement update
#
#*****************************************************************************
 
def update(X_hat_t,P_t,Z_t,R_t,H_t):
    
    # Compute the Kalman Gain
    K_n = calc_kf_gain(P_n_prior, H, R)
    
    # Update the estimate via Zk
    #X_t=X_hat_t+K_prime.dot(Z_t-H_t.dot(X_hat_t))
    X_hat_n = X_hat_n_prior + K_n * (Z_n ) 

    # Update the error covariance
    #P_t=P_t-K_prime.dot(H_t).dot(P_t)
    P_n = (1 - (K_n * H)) * P_n_prior

    return X_hat_n,P_n

#*****************************************************************************
#
# The following stops the program from automatically closing.
#
#*****************************************************************************

# when any key is pressed it will jump to sys.exit() 
input("Press Enter to quit.") 
 
# sys.exit() is used to make the program quits. ( duh ) 
sys.exit() 