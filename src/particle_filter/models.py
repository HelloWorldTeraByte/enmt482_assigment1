################################################################################
#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,                 #
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/                  #
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~                #
################################################################################
#                                                                              #
#                                  models.py                                   #
#                                                                              #
################################################################################
# Authors:        Jason Ui
#                 Randipa Gunathilake
#                 M.P. Hayes and M.J. Edwards,
#                 Department of Electrical and Computer Engineering
#                 University of Canterbury
#
# Date created:       22/08/2021
# Date Last Modified: 31/08/2021
################################################################################
#  Module Description:
#
#  Particle filter sensor and motion model implementations.
#
################################################################################

# Include Standard Modules
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp, pi
from numpy.random import randn, normal
from scipy.stats import norm
import numpy as np

# Application Modules Includes
from particle_filter_extra.utils import gauss, wraptopi, angle_difference

def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """
    # Variable Initialisation
    # Initialise Particle Count
    M = particle_poses.shape[0]

    # Local pose change parameterisation
    local_pose_change_vector = np.zeros((M, 3))
    odom_dy = (odom_pose[1] - odom_pose_prev[1])
    odom_dx = (odom_pose[0] - odom_pose_prev[0])
    rot_before_local = arctan2(odom_dy,odom_dx) - odom_pose_prev[2]
    rot_after_local = odom_pose[2] - odom_pose_prev[2] - rot_before_local
    translation_local = sqrt(odom_dy**2 + odom_dx**2)

    # Appromixate global pose change from local pose change
    local_pose_change_vector = np.array((translation_local, rot_before_local, rot_after_local))
    global_pose_change_vector = local_pose_change_vector

    # Probabilistic Odometry Motion Model Extension
    # Adding Additive Noise
    alpha = np.array((0.01, 0.01, 0.01, 0.01)) 
    translation_std = (alpha[2] * (abs(global_pose_change_vector[1]) + abs(global_pose_change_vector[2]))) + (alpha[3]* global_pose_change_vector[0])
    bearing_before_std = (alpha[0] * abs(global_pose_change_vector[1])) + (alpha[1]* global_pose_change_vector[0])
    bearing_after_std = (alpha[0] * abs(global_pose_change_vector[2])) + (alpha[1]* global_pose_change_vector[0])
    global_pose_change_vector[0] += normal(loc=0, scale=translation_std, size=1)
    global_pose_change_vector[1] += normal(loc=0, scale=bearing_before_std, size=1)
    global_pose_change_vector[2] += normal(loc=0, scale=bearing_after_std, size=1)
    
    # Estimate the new pose from the previous pose
    for m in range(M):

        particle_poses[m, 0] += global_pose_change_vector[0] * cos(particle_poses[m, 2] + global_pose_change_vector[1])
        particle_poses[m, 1] += global_pose_change_vector[0] * sin(particle_poses[m, 2] + global_pose_change_vector[1])
        particle_poses[m, 2] += global_pose_change_vector[1] + global_pose_change_vector[2]
    
    return particle_poses


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """

    # Variable Initialisation
    # Initialise Particle Count and Particle Weight
    # M is an integer-type variable
    # Particle_weights is an array-type varialbe
    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)

    # Convert relative pose measurements into range and bearing
    for m in range(M):
        
        # calcuate particle range and bearing
        x_diff = beacon_loc[0] - particle_poses[m,0]
        y_diff = beacon_loc[1] - particle_poses[m,1]
        particle_range = np.sqrt(x_diff**2 + y_diff**2)
        particle_bearing = angle_difference( particle_poses[m,2], arctan2(y_diff, x_diff))

        # calculate beacon range and bearing
        beacon_range = sqrt(beacon_pose[0]**2 + beacon_pose[1]**2)
        beacon_bearing = arctan2(beacon_pose[1], beacon_pose[0])

        # calculate range and bearing error
        range_error = beacon_range - particle_range
        bearing_error = angle_difference(particle_bearing, beacon_bearing)

        # Set Standard Deviation
        range_error_std =  0.1
        bearing_error_std = 0.1

        # Determine range and bearing error PDF
        range_error_pdf = gauss(range_error,0,range_error_std)
        bearing_error_pdf = gauss(bearing_error,0,bearing_error_std)

        # Update Particle weights
        likelihood = range_error_pdf * bearing_error_pdf
        particle_weights[m] = likelihood

    return particle_weights
