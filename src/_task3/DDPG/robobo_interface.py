import robobo
import numpy as np
from os.path import abspath

# Constants
USED_IR_SENSORS = [0, 2, 3, 5, 7]


class SimulatedRobobo:
    def __init__(self, ip='192.168.1.113', robot_id='#0'):
        self._env = robobo.SimulationRobobo(robot_id).connect(address=ip, port=19997)

    def start(self, randomize_arena=True, hide_render=True):
        """ Start simulation in V-REP
        """
        # Start V-REP simulation (optionally randomize place of food)
        if randomize_arena:
            self._env.randomize_arena_task3()
        self._env.play_simulation()

        # Not rendering saves run time
        if hide_render:
            self._env.toggle_visualization()

        # Set robot to starting position
        self._randomly_rotate()
        self.reset_tilt()

    def _randomly_rotate(self):
        turn_dir = np.random.choice([-8, 8])
        duration = np.random.randint(0, 5000)
        self.move(turn_dir, -turn_dir, duration)

    def reset_tilt(self):
        self._env.set_phone_tilt(0.8, 100)  # tilt it forward

    def stop(self):
        """ Stops simulation in V-REP
        """
        self._env.stop_world()
        self._env.wait_for_stop()

    def take_picture(self):
        """ Takes a picture with the front facing camera
        """
        return self._env.get_image_front()

    def distance_between(self, A, B):
        return self._env.distance_between(A, B)

    def get_sensor_state(self):
        """ Return sensor values limited to specified sensor group
        """
        return [d for i, d in enumerate(self._env.read_irs()) if i in USED_IR_SENSORS]

    def move(self, left, right, millis=1000):
        """ Moves robot by rotating the left and/or right wheels
        """
        self._env.move(left, right, millis)


class HardwareRobobo:
    def __init__(self, ip='192.168.1.113', enable_camera=True):
        self._env = robobo.HardwareRobobo(camera=enable_camera).connect(address=ip)
        self.reset_tilt()

    def reset_tilt(self):
        self._env.set_phone_tilt(109, 100)  # tilt it forward

    def sleep(self, sec):
        """ Halts robot
        """
        self._env.sleep(sec)

    def has_collided(self, prox_max=5000):
        """ Checks whether a collision has occurred (front/rear)
        """
        for i, d in enumerate(self._env.read_irs()):
            if d > prox_max:
                if i < 2:
                    return 'rear'
                else:
                    return 'front'
        return False

    def take_picture(self):
        """ Takes a picture with the front facing camera
        """
        return self._env.get_image_front()

    def get_sensor_state(self):
        """ Return sensor values limited to specified sensor group
        """
        # Optionally, discard values of IR sensors pointing to the floor
        return [d for i, d in enumerate(self._env.read_irs()) if i in USED_IR_SENSORS]

    def move(self, left, right, millis=1000):
        """ Moves robot by rotating the left and/or right wheels
        """
        self._env.move(left, right, millis)
