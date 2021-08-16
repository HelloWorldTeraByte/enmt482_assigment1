import numpy as np
from matplotlib.pyplot import subplots, show, savefig

# Load data
filename = 'training1.csv'
#filename = 'training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split data into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

dt = time[1:] - time[0:-1]
velocity_estimated = np.gradient(distance, time)

fig, axes = subplots(1)
axes.plot(time, velocity_command, label='command speed')
axes.plot(time, velocity_estimated, label='estimated speed')
axes.legend()
axes.set_xlabel('Time')
axes.set_ylabel('Speed (m/s)')

savefig(__file__.replace('.py', '.pdf'))
