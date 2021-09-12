import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.interpolate import splev, splrep
from sklearn.linear_model import RANSACRegressor
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time_data, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

class IR4SensorModel(object):
    def __init__(self, distance, measurement, smooth=130):
        self.distance = distance
        self.measurement = measurement
        self.smooth_val = smooth

        self.plots_init()
        self.calculate()
        self.plots_draw()
    
    def slider_update(self, val):
        self.smooth_val = self.smooth_slider.val
        self.calculate()
        self.plots_draw()

    def calculate(self):
        self.distance_sorted, self.measurement_sorted = zip(*sorted(zip(self.distance, self.measurement)))
        self.spline = splrep(self.distance_sorted, self.measurement_sorted, s=self.smooth_val)

        #self.spline_dist = np.linspace(0.05, 3.5, 500)
        self.spline_meas = splev(self.distance, self.spline)

        self.errors = self.measurement - self.spline_meas
        self.error_var = np.var(self.errors)
        print('Error Variance: ', self.error_var)

    def plots_init(self):
        self.err_fig, self.err_ax = plt.subplots(2)
        plt.title("Errors")

        self.sensor_fig, self.sensor_ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        plt.title("IR4")
        self.sens_ax_smooth = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.smooth_slider = Slider(self.sens_ax_smooth, 'Smoothing', 1, 200, self.smooth_val)
        self.smooth_slider.on_changed(self.slider_update)

    def plots_draw(self):
        self.err_ax[0].clear()
        self.err_ax[1].clear()
        self.err_ax[0].scatter(self.distance, self.errors)
        self.err_ax[1].hist(self.errors, 100)
        self.err_fig.canvas.draw()

        self.sensor_ax.clear()
        self.sensor_ax.plot(self.distance, self.measurement, '.')
        self.sensor_ax.plot(self.distance, self.spline_meas, color='red', linewidth=2)
        self.sensor_fig.canvas.draw()

ir4_sen = IR4SensorModel(distance, raw_ir4)
plt.show()