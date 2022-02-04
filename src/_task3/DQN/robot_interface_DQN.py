import sys
import robobo
import signal
import socket
import numpy as np

class RoboboEnv:
    def __init__(self, env_type='simulation', ip=socket.gethostbyname(socket.gethostname()), robot_id='', 
                    used_sensors='5', camera=True, hide_render=True, task=1):
        signal.signal(signal.SIGINT, self.terminate_program)
        # Init simulated or hardware arena
        if env_type == 'simulation' or env_type == 'randomized_simulation':
            self._env = robobo.SimulationRobobo(robot_id).connect(address=ip, port=19997)
        elif env_type == 'hardware':
            self._env = robobo.HardwareRobobo(camera=camera).connect(address=ip)
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
        self._task = task

        self._env.set_phone_tilt(90, 100)

    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    @property
    def in_simulation(self):
        return self._in_simulation

    def start(self, safe_space, n_objects=8):
        """ Start simulation in V-REP """
        if self._in_simulation:
            # Optionally randomize arena
            if self._randomize_arena:
                if self._task == 1:
                    self._env.randomize_arena(safe_space=safe_space, n_objects=n_objects)
                elif self._task == 2:
                    self._env.randomize_food_margin(safe_space=safe_space, n_objects=n_objects)
                # self._env.randomize_food()
                elif self._task == 3:
                    self._env.randomize_arena_task3(safe_space=safe_space)


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

    def take_action(self, action:int, SIM=True):
        if SIM:
            if action == 0: 
                self.move(5, 15) # Move Left
            elif action == 1: 
                self.move(5, 5) # Move Forwards
            elif action == 2: 
                self.move(15, 5) # Move Right
            else:
                self.move(-5, -5) # Move Backwards
        else:
            if action == 0: 
                self.move(10, 25) # Move Left
            elif action == 1: 
                self.move(15, 5) # Move Forwards
            elif action == 2: 
                self.move(35, 5) # Move Right
            else:
                self.move(-15, -5) # Move Backwards

    def take_picture(self):
        """ Takes a picture with the front facing camera
        """
        return self._env.get_image_front()

    def distance_between(self, A, B):
        return self._env.distance_between(A, B)