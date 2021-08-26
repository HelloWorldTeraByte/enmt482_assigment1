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
#   Date created:       27/08/2021
#   Date Last Modified: 27/08/2021


#************************************************************************/

#  Module Description:
#
#

import numpy as np
from matplotlib.pyplot import subplots, show, savefig
import matplotlib.pyplot as plt
import statistics
import decimal

#*****************************************************************************
#
# The following is to load data
#
#*****************************************************************************

filename = 'part_a_old/training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split data into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

#*****************************************************************************
#
# The following is for class
#
#*****************************************************************************

# TODO
#class Motion_Data_Plotter(object):
#    def set_ticks()


#*****************************************************************************
#
# The following is for functions
#
#*****************************************************************************

def round_down(value, decimals):
    with decimal.localcontext() as ctx:
        d = decimal.Decimal(value)
        ctx.rounding = decimal.ROUND_DOWN
        return round(d, decimals)

#*****************************************************************************
#
# The following is for main program
#
#*****************************************************************************

# Defining Variables
dt = time[1:] - time[0:-1]
velocity_estimated = np.gradient(distance, time)
curr_cmd_velocity = velocity_command[0:-1] # u_n_1
curr_dist = distance[0:-1]                 # x_n_1
next_dist = distance[1:]                   # x_n

# Determining Variables
motion_model = curr_cmd_velocity * dt                               # g = u_n_1 * dt
process_noise = next_dist - curr_dist - motion_model                # w_n = x_n - x_n_1 - g
process_noise_var = statistics.variance(process_noise, xbar=None)   # sigma_squared_w
predicted_dist = curr_dist + motion_model + process_noise           # x_n = x_n_1 + g + w_n

#*****************************************************************************
#
# The following is for graphing plots
#
#*****************************************************************************

fig, axes = subplots(2)
axes[0].plot(time, velocity_command, label='command speed')
axes[0].plot(time, velocity_estimated, label='estimated speed')
axes[0].legend()
axes[0].set_title('Commanded & Estimated Velocity')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Speed (m/s)')
axes[0].grid()

major_ticks_x = np.arange(0, max(time), 10)
minor_ticks_x = np.arange(0, max(time), 2)
major_ticks_y = np.arange(round_down(min(velocity_estimated),1), max(velocity_estimated), 0.2)
minor_ticks_y = np.arange(round_down(min(velocity_estimated),1), max(velocity_estimated), 0.05)

axes[0].set_xticks(major_ticks_x)
axes[0].set_xticks(minor_ticks_x, minor=True)
axes[0].set_yticks(major_ticks_y)
#axes[0].set_yticks(minor_ticks_y, minor=True)

axes[0].grid(which='minor', alpha=0.2)
axes[0].grid(which='major', alpha=0.5)

axes[1].plot(time[0:-1], predicted_dist, color='blue', alpha=0.2)
axes[1].plot(time[0:-1], next_dist, color='blue', alpha=0.2)
axes[1].legend()
axes[1].set_title('Estimated Distance')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Distance (m)')

# Handling Grid Lines
major_ticks_x = np.arange(0, max(time), 10)
minor_ticks_x = np.arange(0, max(time), 2)
major_ticks_y = np.arange(0, 4, 0.5)
minor_ticks_y = np.arange(0, 4, 0.2)

axes[1].set_xticks(major_ticks_x)
axes[1].set_xticks(minor_ticks_x, minor=True)
axes[1].set_yticks(major_ticks_y)
#axes[1].set_yticks(minor_ticks_y, minor=True)

axes[1].grid(which='minor', alpha=0.2)
axes[1].grid(which='major', alpha=0.5)

fig2 = plt.figure(2)
plt.hist(process_noise, bins=200, density=True)
plt.title('Histogram of Errors between Measured Data and Model') 
plt.ylabel('Count')
plt.xlabel('Measurement Error')

show()
savefig(__file__.replace('.py', '.pdf'))
