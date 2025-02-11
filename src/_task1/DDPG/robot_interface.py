import robobo
import numpy as np


class RoboboEnv:
    def __init__(self, env_type='simulation', ip='192.168.1.113', robot_id='#0', used_sensors='5', hide_render=True):
        # Init simulated or hardware arena
        if env_type == 'simulation' or env_type == 'randomized_simulation':
            self._env = robobo.SimulationRobobo(robot_id).connect(address=ip, port=19997)
        elif env_type == 'hardware':
            self._env = robobo.HardwareRobobo(camera=False).connect(address=ip)
        else:
            raise Exception('env_type %s not supported' % env_type)

        # What IR sensors are available?
        if used_sensors == '8':
            self._sensors = [0, 1, 2, 3, 4, 5, 6, 7]
        elif used_sensors == '5':
            self._sensors = [0, 2, 3, 5, 7]  # remove down-pointing sensors
        else:
            raise Exception('used_sensors=%s not supported' % used_sensors)

        # Parameters
        self._randomize_arena = env_type.startswith('randomized')
        self._in_simulation = env_type.endswith('simulation')
        self._hide_render = hide_render

    @property
    def in_simulation(self):
        return self._in_simulation

    def start(self):
        """ Start simulation in V-REP """
        if self._in_simulation:
            # Optionally randomize arena
            if self._randomize_arena:
                self._env.randomize_arena()

            # Start simulation in V-REP
            self._env.play_simulation()

            # Not rendering saves run time
            if self._hide_render:
                self._env.toggle_visualization()

    def sleep(self, sec):
        self._env.sleep(sec)

    def stop(self):
        """ Stops simulation in V-REP """
        if self._in_simulation:
            self._env.stop_world()
            self._env.wait_for_stop()

    @property
    def position(self):
        return self._env.position()

    def has_collided(self, d_min=0.04, prox_max=5000):
        """ Checks whether a collision has occurred and where (front/rear). """
        for i, d in enumerate(self._env.read_irs()):
            if self.in_simulation:
                if type(d) != bool and d < d_min:
                    return True
            else:
                if d > prox_max:
                    return True
        return False

    def get_sensor_state(self):
        """ Return sensor values limited to specified sensor group """
        # Optionally, discard values of IR sensors pointing to the floor
        return [d for i, d in enumerate(self._env.read_irs()) if i in self._sensors]

    def move(self, left, right, millis=1000):
        """ Moves robot by rotating the left and/or right wheels """
        self._env.move(left, right, millis)
