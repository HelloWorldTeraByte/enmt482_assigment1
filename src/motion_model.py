#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~
#************************************************************************/
#*                                                                      */
#                            motion_model.py
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
from matplotlib.pyplot import subplots, show
import numpy as np

# Load data ir3
filename = 'part_a/training2.csv'
data_t2 = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time_t2, distance_t2, velocity_command_t2, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data_t2.T

filename = 'part_a/training1.csv'
data_t1 = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time_t1, distance_t1, velocity_command_t1, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data_t1.T

filename = 'part_a/calibration.csv'
data_c = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time_c, distance_c, velocity_command_c, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1_c, sonar2 = data_c.T

row_num = 2
col_num = 3
fig, axes = subplots(row_num, col_num)
fig.suptitle('Velocity')

#Est Speed
est_vel_c = []
ii= 0
for ii in range(0, len(time_c)):
    est_vel_c.append(distance_c[ii]/time_c[ii])

iii = 0
est_vel_t2 = []
for iii in range(0, len(time_t2)):
    est_vel_t2.append(distance_t2[iii]/time_t2[iii])

iiii = 0
est_vel_t1 = []
for iiii in range(0, len(time_t1)):
    est_vel_t1.append(distance_t1[iiii]/time_t1[iiii])


axes[0, 0].plot(time_c, velocity_command_c, color='black', alpha=1)
#axes[0, 0].plot(time_c, distance_c, color='red', alpha=1)
axes[0, 0].plot(time_c, est_vel_c, color='blue', alpha=1)
axes[0, 0].set_title('Calibration')

axes[0, 1].plot(time_t1, velocity_command_t1,  alpha=0.2)
axes[0, 1].plot(time_t1, est_vel_t1, color='red', alpha=0.2)
axes[0, 1].set_title('Training1')

axes[0, 2].plot(time_t2, velocity_command_t2, alpha=0.2)
axes[0, 2].plot(time_t2, est_vel_t2, color='red', alpha=0.2)
axes[0, 2].set_title('Training2')

# axes[1, 0].plot(distance, raw_ir4, '.',markersize=2, alpha=0.2)
# axes[1, 0].set_title('IR4')

# axes[1, 1].plot(distance, sonar1, '.',markersize=2 , alpha=0.2)
# axes[1, 1].set_title('Sonar1')

# axes[1, 2].plot(distance, sonar2, '.',markersize=2 , alpha=0.2)
# axes[1, 2].set_title('Sonar2')

i = 0
j = 0
for i in range(0,row_num):
    for j in range(0,col_num):
        axes[i,j].set_ylabel('Velocity')

show()