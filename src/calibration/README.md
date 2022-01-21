In this folder you can find the code used to calibrate the Robobo hardware (i.e. IR-sensor measurements, motors, etc.).

## Usage

### Acquire simulation data
1. Open V-REP and load `calibration_scene.ttt`
2. In `collect_calibration_data.py` set `ENV_TYPE='simulation'` and change `IP_ADDR` to current IP
3. Run `py -3 collect_calibration_data.py`

A simulation should start in V-REP, after which a file `sensor_calib_simulation.out` will be created containing in-simulation sensor measurements.

### Acquire hardware data
1. Connect to the Robobo robot
2. In `collect_calibration_data.py` set `ENV_TYPE='hardware'` and `IP_ADDR` to IP of the robot
3. Place robot 1x its own length away from a wall (preferably white like the arena). Make sure the robot faces the wall head-on.
4. Run `py -3 collect_calibration_data.py`

The Robobo should start to move forward, after which a file `sensor_calib_hardware.out` will be created containing sensory measurements from the robot.

### Analyze the results
1. Run `py -3 calibrate_robobo.py`
2. Check plot to make sure mapping was succesful :)

![alt text](https://i.imgur.com/4f9yoaC.png)

If all went well, a file `calib_params.out` is created. This file contains model parameters to correct the (hardware) IR-sensor measurements and a constant to correct for speed differences (the _duration multiplier_).

### Use in our code
```
from calibration.calibrate_robobo import Calibrator

...
hardware_ir_values = hardware_env.read_irs()

calibration = Calibrator('calib_params.out')
duration_multiplier = calibration.duration_multiplier
corrected_ir_values = calibration.correct_sensors(hardware_ir_values)
```
