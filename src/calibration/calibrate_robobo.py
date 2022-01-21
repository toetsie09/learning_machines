import os
import numpy as np
import matplotlib.pyplot as plt


class Calibrator:
    """ Implements a calibration model (y = a*x + b*x^2 + c*e^x + d) to
        correct sensory IR measurements of the Robobo robot to match the
        simulated distance values the robot is trained on.
    """
    def __init__(self, param_file=''):
        self._multip = 1.0
        self._params = None

        if param_file != '' and os.path.isfile(param_file):
            data = np.loadtxt(param_file)
            self._multip = data[0]
            self._params = data[1:]

    @property
    def duration_multiplier(self):
        return self._multip

    @staticmethod
    def _to_data_matrix(x):
        """ Transforms a sequence of values into a data matrix.

            args
            list x: sensor measurements obtained from hardware

            returns: ndarray
        """
        ones = np.ones(len(x))
        return np.array([x, x ** 2, x ** 3, np.sqrt(x), ones]).T

    def fit(self, simulated_dists, hardware_dists):
        """ Fits the model to the hardware sensory data to map them to the
            simulated sensor range and computes the duration multiplier.

            args
            list simulated_dists: List of simulated sensor measurements
            list hardware_dists:  List of hardware sensor measurements

            returns: None
        """
        # Take average of samples within step
        simulated_dists = np.mean(simulated_dists, axis=1)
        hardware_dists = np.mean(hardware_dists, axis=1)

        # Compute duration multiplier to speed/slow-down hardware robot
        self._multip = len(hardware_dists) / len(simulated_dists)
        print("\nDuration multiplier:", self._multip)

        # Resample hardware measurements to the same number of steps as the simulator
        # (which assumes that, in the end, the same distance was traveled)
        interp_hardware_dists = np.interp(np.linspace(0, 1, len(simulated_dists)),
                                          np.linspace(0, 1, len(hardware_dists)),
                                          hardware_dists)

        # Fit polynomial to map from hardware signal to simulated signal
        X = self._to_data_matrix(interp_hardware_dists)

        y = np.array(simulated_dists)
        self._params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # (X^T.X)^-1.X^T.y
        print("Mapping parameters:", self._params)

    def save(self, path):
        """ Saves the model parameters and duration multiplier to a file

            args
            str path: filename of the savefile
        """
        if self._params is None:
            raise Exception('Lets fit the calibration model first shall we?')

        values = np.array([self._multip] + list(self._params.flatten()))
        np.savetxt(path, values)

    def correct_sensors(self, hardware_values):
        """ Takes the raw sensor values from the hardware and corrects it to the simulated
            sensor range the robot was trained with (i.e. a linear function in 0.0 to 0.2).

            args
            hardware_values: hardware sensor values obtained through env.read_irs()

            returns: corrected ir-sensor values
        """
        X = self._to_data_matrix(np.array(hardware_values))
        return X.dot(self._params)


if __name__ == "__main__":
    # Calibration data obtained with collect_calibration_data.py
    simulated_dists = np.loadtxt('sensor_calib_simulation.out')
    hardware_dists = np.loadtxt('sensor_calib_hardware.out')

    # Create mapping
    c = Calibrator()
    c.fit(simulated_dists, hardware_dists)
    c.save('calib_params.out')

    corrected_dists = c.correct_sensors(np.mean(hardware_dists, axis=1))

    # Plot results
    plt.plot(np.linspace(0, 1, len(simulated_dists)), np.mean(simulated_dists, axis=1)[::-1], c='C4', label='simulated')
    plt.plot(np.linspace(0, 1, len(hardware_dists)), np.mean(hardware_dists, axis=1)[::-1], c='C0', label='hardware')
    plt.plot(np.linspace(0, 1, len(corrected_dists)), corrected_dists[::-1], c='C1', label='corrected hardware')
    plt.xlabel('Actual distance to obstacle (base lengths)')
    plt.ylabel('Sensor measurement (a.u.)')
    plt.legend()
    plt.show()

