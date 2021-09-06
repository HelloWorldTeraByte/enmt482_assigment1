"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp, pi
from numpy.random import randn, normal
from utils import gauss, wraptopi, angle_difference
from scipy.stats import norm


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

    M = particle_poses.shape[0]

    #global_pose_change_joint_PDF = norm.pdf(translation_local) * norm.pdf(rot_before_local) * norm.pdf(rot_after_local)

    # TODO.  For each particle calculate its predicted pose plus some
    # additive error to represent the process noise.  With this demo
    # code, the particles move in the -y direction with some Gaussian
    # additive noise in the x direction.  Hint, to start with do not
    # add much noise.

    local_pose_change_vector = np.zeros((M, 3))

    odom_pose_change_y = (odom_pose[1] - odom_pose_prev[1])
    odom_pose_change_x = odom_pose[0] - odom_pose_prev[0]
    
    rot_before_local = np.arctan2(odom_pose_change_y,odom_pose_change_x) - odom_pose_prev[2]
    rot_after_local = odom_pose[2] - odom_pose_prev[2] - rot_before_local
    translation_local = sqrt(odom_pose_change_y**2 + odom_pose_change_x**2)

    local_pose_change_vector = np.array((translation_local, rot_before_local, rot_after_local))
    global_pose_change_vector = local_pose_change_vector #.transpose()

    # global_pose_change_PDF = normal(loc=local_pose_change_vector[1], scale=local_pose_change_vector[0], size=1) \
    # * normal(loc=global_pose_change_vector[0], scale=local_pose_change_vector[0], size=1) \
    # * normal(loc=global_pose_change_vector[2], scale=local_pose_change_vector[0], size=1)

    # Probabilisitc Odometry Motion Sensor
    # (40, 0.1, 0.01, 1) - follows path but soon loses path
    # (5, 20, 50, 0.1) - very bad
    # (80, 1, 0.1, 5) - follows path but fizzling
    # (40, 1, 0.1, 1) - similar to first but broken
    # (50, 0.2, 0.02, 2)
    alpha = np.array((50, 0.2, 0.02, 2))

    translation_std = (alpha[2] * (abs(global_pose_change_vector[1]) + abs(global_pose_change_vector[2]))) + (alpha[3]* global_pose_change_vector[0])
    bearing_before_std = (alpha[0] * abs(global_pose_change_vector[1])) + (alpha[1]* global_pose_change_vector[0])
    bearing_after_std = (alpha[0] * abs(global_pose_change_vector[2])) + (alpha[1]* global_pose_change_vector[0])

    global_pose_change_vector[0] = normal(loc=global_pose_change_vector[0], scale=translation_std, size=1)
    global_pose_change_vector[1] = normal(loc=global_pose_change_vector[1], scale=bearing_before_std, size=1)
    global_pose_change_vector[2] = normal(loc=global_pose_change_vector[2], scale=bearing_after_std, size=1)

    for m in range(M):
        particle_poses[m, 0] += global_pose_change_vector[0] * cos(odom_pose_prev[2] + global_pose_change_vector[1])
        particle_poses[m, 1] += global_pose_change_vector[0] * sin(odom_pose_prev[2] + global_pose_change_vector[1])
        particle_poses[m, 2] += global_pose_change_vector[1] + global_pose_change_vector[2]

        # particle_poses[m, 0] += randn(1) * 0.1
        # particle_poses[m, 1] -= 0.1
    
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

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)
    
    # TODO.  For each particle calculate its weight based on its pose,
    # the relative beacon pose, and the beacon location.

    for m in range(M):
        
        particle_weights[m] = 1

        x_diff = beacon_loc[0] - particle_poses[m,0]
        y_diff = beacon_loc[1] - particle_poses[m,1]
        particle_range = sqrt(x_diff**2 + y_diff**2)
        particle_bearing = angdiff(arctan2(y_diff, x_diff), particle_poses[m,2])

        beacon_range = sqrt(beacon_pose[0]**2 + beacon_pose[1]**2)
        beacon_bearing = arctan2(beacon_pose[1], beacon_pose[0])

        range_error_pdf = norm.pdf(beacon_range - particle_range)
        bearing_error_pdf = norm.pdf(angdiff(beacon_bearing, particle_bearing))

        particle_weights[m] *= range_error_pdf * bearing_error_pdf

    
    return particle_weights
