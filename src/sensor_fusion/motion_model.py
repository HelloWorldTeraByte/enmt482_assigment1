################################################################################
#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,                 #
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/                  #
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~                #
################################################################################
#                                                                              #
#                            motion_model.py                                   #
#                                                                              #
################################################################################
# Authors:        Jason Ui
#                 Randipa Gunathilake
#
# Date created:       27/08/2021
# Date Last Modified: 27/08/2021
################################################################################
#  Module Description:
#
#
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import errno

################################################################################
#
# The following is for motion model class
#
################################################################################
class MotionModel(object):
    def __init__(self, distance = [], time = [], velocity_command = [], use_saved = True, training_no = 1, plot = False):
        try:
            os.mkdir('data')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.pickle_loc = 'data/motion_model.pckl'

        if(use_saved):
            with open(self.pickle_loc, "rb") as f:
                self.process_noise_var = pickle.load(f)
                return

        self.distance = distance
        self.time = time
        self.velocity_cmd = velocity_command
        self.velocity_est = np.gradient(distance, time)
        self.next_dist = self.distance[1:]                      # x_n
        self.curr_dist = self.distance[0:-1]                    # x_n_1
        self.dt = self.time[1:] - self.time[0:-1]
        self.curr_cmd_velocity = self.velocity_cmd[0:-1]    # u_n_1
        self.motion_model = self.curr_cmd_velocity * self.dt    # g = u_n_1 * dt
        self.process_noise = self.next_dist - self.curr_dist - self.motion_model # w_n = x_n - x_n_1 - g
        self.process_noise_var = np.var(self.process_noise)   # sigma_squared_w
        #self.predicted_dist = self.curr_dist + self.motion_model + self.process_noise # x_n = x_n_1 + g + w_n

        self.predicted_dist = np.zeros(np.size(self.time))
        for i in range(1, np.size(self.time)):
            self.predicted_dist[i] = self.predicted_dist[i-1] + self.velocity_cmd[i] * (time[i] - time[i-1])

        self.training_no = training_no

        pickle_data = self.process_noise_var
        with open(self.pickle_loc, "wb") as f:
            pickle.dump(pickle_data, f)
 
        # Handle plotting
        if(plot):
            self.plots_init()
            self.plots_grid_init()
            self.plot_hist()
    
    def get_variance(self):
        return self.process_noise_var

    def plots_init(self):
        self.fig, self.axes = plt.subplots(2)
        self.axes[0].plot(self.time, self.velocity_cmd, label='command speed')
        self.axes[0].plot(self.time, self.velocity_est, label='estimated speed')
        self.axes[0].legend()
        self.axes[0].set_title('Commanded & Estimated Velocity')
        self.axes[0].set_xlabel('Time (s)')
        self.axes[0].set_ylabel('Speed (m/s)')

        self.axes[1].plot(self.time, self.predicted_dist, label='Estimated Distance')
        self.axes[1].plot(self.time, self.distance, label='True Distance')
        self.axes[1].set_title('Estimated Distance')
        self.axes[1].set_xlabel('Time (s)')
        self.axes[1].set_ylabel('Distance (m)')
        
    def plots_grid_init(self):
        self.axes[0].grid()
        if self.training_no == 1:
            major_ticks_x = np.arange(0, max(self.time), 50)
            minor_ticks_x = np.arange(0, max(self.time), 10)
            major_ticks_y = np.arange(round(min(self.velocity_est),1), max(self.velocity_est), 0.2)
            minor_ticks_y = np.arange(round(min(self.velocity_est),1), max(self.velocity_est), 0.05)
        if self.training_no == 2:
            major_ticks_x = np.arange(0, max(self.time), 10)
            minor_ticks_x = np.arange(0, max(self.time), 2)
            major_ticks_y = np.arange(round(min(self.velocity_est),1), max(self.velocity_est), 0.2)
            minor_ticks_y = np.arange(round(min(self.velocity_est),1), max(self.velocity_est), 0.05)

        self.axes[0].set_xticks(major_ticks_x)
        self.axes[0].set_xticks(minor_ticks_x, minor=True)
        self.axes[0].set_yticks(major_ticks_y)
        self.axes[0].set_yticks(minor_ticks_y, minor=True)
        self.axes[0].grid(which='minor', alpha=0.2)
        self.axes[0].grid(which='major', alpha=0.5)

        if self.training_no == 1: 
            major_ticks_x = np.arange(0, max(self.time), 50)
            minor_ticks_x = np.arange(0, max(self.time), 10)
            major_ticks_y = np.arange(0, 4, 0.5)
            minor_ticks_y = np.arange(0, 4, 0.2)

        if self.training_no == 2:
            major_ticks_x = np.arange(0, max(self.time), 10)
            minor_ticks_x = np.arange(0, max(self.time), 2)
            major_ticks_y = np.arange(0, 4, 0.5)
            minor_ticks_y = np.arange(0, 4, 0.2)

        self.axes[1].set_xticks(major_ticks_x)
        self.axes[1].set_xticks(minor_ticks_x, minor=True)
        self.axes[1].set_yticks(major_ticks_y)
        self.axes[1].set_yticks(minor_ticks_y, minor=True)

        self.axes[1].grid(which='minor', alpha=0.2)
        self.axes[1].grid(which='major', alpha=0.5)

    def plot_hist(self):
        self.fig2 = plt.figure(2)
        plt.hist(self.process_noise, bins=200, density=True, label='training' + str(self.training_no))
        plt.title('Histogram of Errors between Measured Data and Model') 
        plt.ylabel('Count')
        plt.xlabel('Measurement Error')
        plt.legend()
        

if __name__ == "__main__":
    file_c = '../../res/sensor_fusion/calibration.csv'
    data_c = np.loadtxt(file_c, delimiter=',', skiprows=1)
    index_c, time_c, distance_c, velocity_command_c, raw_ir1_c, raw_ir2_c, raw_ir3_c, raw_ir4_c, \
        sonar1_c, sonar2_c = data_c.T

    file_t1 = '../../res/sensor_fusion/training1.csv'
    data_t1 = np.loadtxt(file_t1, delimiter=',', skiprows=1)
    index_t1, time_t1, distance_t1, velocity_command_t1, raw_ir1_t1, raw_ir2_t1, raw_ir3_t1, raw_ir4_t1, \
        sonar1_t1, sonar2_t1 = data_t1.T

    file_t2 = '../../res/sensor_fusion/training2.csv'
    data_t2 = np.loadtxt(file_t2, delimiter=',', skiprows=1)
    index_t2, time_t2, distance_t2, velocity_command_t2, raw_ir1_t2, raw_ir2_t2, raw_ir3_t2, raw_ir4_t2, \
        sonar1_t2, sonar2_t2 = data_t2.T

    motion_model_plot_c = MotionModel(distance_c, time_c, velocity_command_c, use_saved=False, training_no=1, plot=True)
    motion_model_plot_t1 = MotionModel(distance_t1, time_t1, velocity_command_t1, use_saved=False, training_no=1, plot=True)
    motion_model_plot_t2 = MotionModel(distance_t2, time_t2, velocity_command_t2, use_saved=False, training_no=2, plot=True)

    print("Calibration Process Variance: ", motion_model_plot_c.process_noise_var)
    print("Training 1 Process Variance: ", motion_model_plot_t1.process_noise_var)
    print("Training 2 Process Variance: ", motion_model_plot_t2.process_noise_var)

    plt.show()