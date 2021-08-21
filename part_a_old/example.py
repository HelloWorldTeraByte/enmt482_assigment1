#!/usr/bin/env python3
"""Example code for ENMT482 assignment."""

import numpy as np
import matplotlib.pyplot as plt


# Load data
filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T


# Plot true range and sonar measurements over time
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(time, range_)
plt.xlabel('Time (s)')
plt.ylabel('Range (m)')
plt.title('True range')

plt.subplot(132)
plt.plot(time, sonar1, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar1')
plt.xlabel('Time (s)')

plt.subplot(133)
plt.plot(time, sonar2, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar2')
plt.xlabel('Time (s)')


# Plot sonar error
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(time, range_ - sonar1, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Sonar1 error')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')

plt.subplot(122)
plt.plot(time, range_ - sonar2, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Sonar2 error')
plt.xlabel('Time (s)')


# Plot IR sensor measurements
plt.figure(figsize=(8, 7))

plt.subplot(221)
plt.plot(range_, raw_ir1, '.', alpha=0.5)
plt.title('IR1')
plt.ylabel('Measurement (V)')

plt.subplot(222)
plt.plot(range_, raw_ir2, '.', alpha=0.5)
plt.title('IR2')

plt.subplot(223)
plt.plot(range_, raw_ir3, '.', alpha=0.5)
plt.title('IR3')
plt.xlabel('Range (m)')
plt.ylabel('Measurement (V)')

plt.subplot(224)
plt.plot(range_, raw_ir4, '.', alpha=0.5)
plt.title('IR4')
plt.xlabel('Range (m)')
plt.show()
