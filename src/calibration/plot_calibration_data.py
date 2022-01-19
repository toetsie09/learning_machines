import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    simulated_dists = np.loadtxt('sensor_calib_simulation.out')[:-1]
    hardware_dists = np.loadtxt('sensor_calib_hardware.out')[:-1]

    simulated_x = np.linspace(0, 1, len(simulated_dists))
    for x, ys in zip(simulated_x, simulated_dists):
        plt.scatter([x] * len(ys), ys * 8000, c='C0')

    hardware_x = np.linspace(0, 1, len(hardware_dists))
    for x, ys in zip(hardware_x, hardware_dists):
        plt.scatter([x] * len(ys), ys, c='C1')

    plt.show()
