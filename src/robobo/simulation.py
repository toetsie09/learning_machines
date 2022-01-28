from __future__ import unicode_literals, print_function, absolute_import, division, generators, nested_scopes
from robobo.base import Robobo
import time
import vrep
import cv2
import numpy as np
from tqdm import tqdm
class VREPCommunicationError(Exception):
    pass

class SimulationRobobo(Robobo):
    def __init__(self, number=""):
        self._clientID = None
        self._value_number = number
        
        self._foods = [] # maintain locations of foods in scene

    def connect(self, address='127.0.0.1', port=19999):
        # vrep.simxFinish(-1)  # just in case, close all opened connections
        self._clientID = vrep.simxStart(address, port, True, True, 5000, 5)  # Connect to V-REP
        if self._clientID >= 0: #  and clientID_0 != -1:
            self.wait_for_ping()
            print('Connected to remote API server: client id {}'.format(self._clientID))
        else:
            raise VREPCommunicationError('Failed connecting to remote API server')

        get_handles_timeout = 120.0

        startTime = time.time()
        while time.time() - startTime < get_handles_timeout:
            try:
                self.initialize_handles()
                return self
            except vrep.VrepApiError as _e:
                print("Handle initialization failed, retrying.")
                time.sleep(0.05)

        return False

    def disconnect(self):
        vrep.unwrap_vrep(
            vrep.simxFinish(self._clientID)
        )
        
    def load_scene(self, filename):
        vrep.unwrap_vrep(
            vrep.simxLoadScene(self._clientID, filename, 0xFF, vrep.simx_opmode_blocking)
        )

    def close_scene(self):
        vrep.unwrap_vrep(
            vrep.simxCloseScene(self._clientID, vrep.simx_opmode_blocking)
        )

    def initialize_handles(self):
        self._RightMotor = self._vrep_get_object_handle('Right_Motor{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._LeftMotor = self._vrep_get_object_handle('Left_Motor{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._Robobo = self._vrep_get_object_handle('Robobo{}'.format(self._value_number), vrep.simx_opmode_blocking)

        self._IrBackC = self._vrep_get_object_handle('Ir_Back_C{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrFrontC = self._vrep_get_object_handle('Ir_Front_C{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrFrontLL = self._vrep_get_object_handle('Ir_Front_LL{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrFrontRR = self._vrep_get_object_handle('Ir_Front_RR{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrBackL = self._vrep_get_object_handle('Ir_Back_L{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrBackLFloor = self._vrep_get_object_handle('Ir_Back_L_Floor{}'.format(self._value_number),
                                                            vrep.simx_opmode_blocking)
        self._IrBackR = self._vrep_get_object_handle('Ir_Back_R{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrBackRFloor = self._vrep_get_object_handle('Ir_Back_R_Floor{}'.format(self._value_number),
                                                            vrep.simx_opmode_blocking)
        self._IrFrontL = self._vrep_get_object_handle('Ir_Front_L{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrFrontLFloor = self._vrep_get_object_handle('Ir_Front_L_Floor{}'.format(self._value_number),
                                                            vrep.simx_opmode_blocking)
        self._IrFrontR = self._vrep_get_object_handle('Ir_Front_R{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._IrFrontRFloor = self._vrep_get_object_handle('Ir_Front_R_Floor{}'.format(self._value_number),
                                                            vrep.simx_opmode_blocking)
        self._TiltMotor = self._vrep_get_object_handle('Tilt_Motor{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._PanMotor = self._vrep_get_object_handle('Pan_Motor{}'.format(self._value_number), vrep.simx_opmode_blocking)
        self._FrontalCamera = self._vrep_get_object_handle('Frontal_Camera{}'.format(self._value_number), vrep.simx_opmode_blocking)

        # read a first value in streaming mode
        self._vrep_read_proximity_sensor_ignore_error(self._IrFrontC)
        self._vrep_read_proximity_sensor_ignore_error(self._IrBackC)
        self._vrep_read_proximity_sensor_ignore_error(self._IrFrontLL)
        self._vrep_read_proximity_sensor_ignore_error(self._IrFrontRR)
        self._vrep_read_proximity_sensor_ignore_error(self._IrBackL)
        self._vrep_read_proximity_sensor_ignore_error(self._IrBackLFloor)
        self._vrep_read_proximity_sensor_ignore_error(self._IrBackR)
        self._vrep_read_proximity_sensor_ignore_error(self._IrBackRFloor)
        self._vrep_read_proximity_sensor_ignore_error(self._IrFrontR)
        self._vrep_read_proximity_sensor_ignore_error(self._IrFrontRFloor)
        self._vrep_read_proximity_sensor_ignore_error(self._IrFrontL)
        self._vrep_read_proximity_sensor_ignore_error(self._IrFrontLFloor)

        # setup join positions
        vrep.simxGetJointPosition(self._clientID, self._RightMotor, vrep.simx_opmode_buffer)
        vrep.simxGetJointPosition(self._clientID, self._LeftMotor, vrep.simx_opmode_buffer)
        vrep.simxGetObjectPosition(self._clientID, self._Robobo, -1, vrep.simx_opmode_buffer)

        # read a first value in buffer mode
        self._vrep_get_vision_sensor_image_ignore_error(self._FrontalCamera, vrep.simx_opmode_streaming)

        self.wait_for_ping()

    def sleep(self, seconds):
        duration = seconds * 1000
        start_time = self.get_sim_time()
        while self.get_sim_time() - start_time < duration:
            pass

    def wait_for_ping(self, timeout_seconds=120.0):
        startTime = time.time()
        while time.time() - startTime < timeout_seconds:
            try:
                self._vrep_get_ping_time()
                # print("check success")
                return True
            except vrep.VrepApiError as _e:
                # print("check failed")
                time.sleep(0.05)
        
        print("{} seconds passed with ping not coming online, you may expericence problems with the connection".format(timeout_seconds))
        return False

    def _vrep_get_ping_time(self):
        return vrep.unwrap_vrep(vrep.simxGetPingTime(self._clientID))

    def _vrep_get_object_handle(self, name, opmode):
        return vrep.unwrap_vrep(vrep.simxGetObjectHandle(self._clientID, name, opmode))

    def _vrep_read_proximity_sensor(self, handle, opmode=vrep.simx_opmode_streaming):
        return vrep.unwrap_vrep(vrep.simxReadProximitySensor(self._clientID, handle, opmode))

    def _vrep_read_proximity_sensor_ignore_error(self, handle, opmode=vrep.simx_opmode_streaming):
        try:
            self._vrep_read_proximity_sensor(handle, opmode)
        except vrep.error.VrepApiError as error:
            if error.ret_code is not vrep.simx_return_novalue_flag:
                raise
        
    def _vrep_get_vision_sensor_image(self, camera_handle, opmode=vrep.simx_opmode_buffer, a=0):
        return vrep.unwrap_vrep(vrep.simxGetVisionSensorImage(self._clientID, camera_handle, a, opmode))

    def _vrep_get_vision_sensor_image_ignore_error(self, camera_handle, opmode=vrep.simx_opmode_buffer, a=0):
        try:
            self._vrep_get_vision_sensor_image(camera_handle, opmode, a)
        except vrep.error.VrepApiError as error:
            if error.ret_code is not vrep.simx_return_novalue_flag:
                raise

    def _vrep_set_joint_target_velocity(self, handle, speed, opmode):
        return vrep.unwrap_vrep(vrep.simxSetJointTargetVelocity(self._clientID, handle, speed, opmode))

    def _vrep_set_joint_target_position(self, handle, position, opmode=vrep.simx_opmode_oneshot):
        return vrep.unwrap_vrep(vrep.simxSetJointTargetPosition(self._clientID, handle, position, opmode))

    def spin(self):
        raise NotImplementedError("Not implemeted yet")

    def set_emotion(self, emotion):
        print("ROBOT EMOTION: {}".format(emotion))

    def move(self, left, right, millis=500):
        normalizer = 10.0
        left = left/normalizer
        right = right/normalizer

        self._vrep_set_joint_target_velocity(self._LeftMotor, left, vrep.simx_opmode_oneshot)
        self._vrep_set_joint_target_velocity(self._RightMotor, right, vrep.simx_opmode_oneshot)
        # self.wait_for_ping() #get_sim_time is already waiting for ping

        duration = millis #/ 1000.0
        # startTime = time.time()
        # while time.time() - startTime < duration:
        #     # rightMotorAngPos = vrep.unwrap_vrep(vrep.simxGetJointPosition(self._clientID, self._RightMotor, vrep.simx_opmode_blocking))
        #     # leftMotorAngPos  = vrep.unwrap_vrep(vrep.simxGetJointPosition(self._clientID, self._LeftMotor, vrep.simx_opmode_blocking))
        #     # RoboAbsPos       = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self._clientID, self._Robobo, -1, vrep.simx_opmode_blocking))
        #     time.sleep(0.005)
        # print("sleeping for {}".format(duration))

        # busy waiting
        start_time = self.get_sim_time()
        while self.get_sim_time() - start_time < duration:
            pass
        
        # Stop to move the wheels motor. Angular velocity.
        stopRightVelocity = stopLeftVelocity = 0
        self._vrep_set_joint_target_velocity(self._LeftMotor, stopLeftVelocity,
                                                  vrep.simx_opmode_oneshot)
        self._vrep_set_joint_target_velocity(self._RightMotor, stopRightVelocity,
                                                  vrep.simx_opmode_oneshot)
        self.wait_for_ping()

    def talk(self, message):
        print("ROBOT SAYS: {}".format(message))

    def set_led(self, selector, color):
        raise NotImplementedError("Not implemeted yet")
    
    def read_irs(self):
        """
        returns sensor readings: [backR, backC, backL, frontRR, frontR, frontC, frontL, frontLL]
        """      
        detectionStateIrFrontC, detectedPointIrFrontC, detectedObjectHandleIrFrontC, \
        detectedSurfaceNormalVectorIrFrontC = self._vrep_read_proximity_sensor(
            self._IrFrontC, vrep.simx_opmode_buffer)
        detectionStateIrBackC, detectedPointIrIrBackC, detectedObjectHandleIrBackC, \
        detectedSurfaceNormalVectorIrBackC = self._vrep_read_proximity_sensor(
            self._IrBackC, vrep.simx_opmode_buffer)
        detectionStateIrFrontLL, detectedPointIrFrontLL, detectedObjectHandleIrFrontLL, \
        detectedSurfaceNormalVectorIrFrontLL = self._vrep_read_proximity_sensor(
            self._IrFrontLL, vrep.simx_opmode_buffer)
        detectionStateIrFrontRR, detectedPointIrFrontRR, detectedObjectHandleIrFrontRR, \
        detectedSurfaceNormalVectorIrFrontRR = self._vrep_read_proximity_sensor(
            self._IrFrontRR, vrep.simx_opmode_buffer)
        detectionStateIrBackL, detectedPointIrBackL, detectedObjectHandleIrBackL, \
        detectedSurfaceNormalVectorIrBackL = self._vrep_read_proximity_sensor(
            self._IrBackL, vrep.simx_opmode_buffer)

        detectionStateIrBackR, detectedPointIrBackR, detectedObjectHandleIrBackR, \
        detectedSurfaceNormalVectorIrBackR = self._vrep_read_proximity_sensor(
            self._IrBackR, vrep.simx_opmode_buffer)

        detectionStateIrFrontR, detectedPointIrFrontR, detectedObjectHandleIrFrontR, \
        detectedSurfaceNormalVectorIrFrontR = self._vrep_read_proximity_sensor(
            self._IrFrontR, vrep.simx_opmode_buffer)

        detectionStateIrFrontL, detectedPointIrFrontL, detectedObjectHandleIrFrontL, \
        detectedSurfaceNormalVectorIrFrontL = self._vrep_read_proximity_sensor(
            self._IrFrontL, vrep.simx_opmode_buffer)

        vect = [np.sqrt(detectedPointIrBackR[0]   ** 2 + detectedPointIrBackR[1]   ** 2 + detectedPointIrBackR[2]   ** 2)
                if detectionStateIrBackR   else False,
                np.sqrt(detectedPointIrIrBackC[0] ** 2 + detectedPointIrIrBackC[1] ** 2 + detectedPointIrIrBackC[2] ** 2)
                if detectionStateIrBackC   else False,
                np.sqrt(detectedPointIrBackL[0] ** 2   + detectedPointIrBackL[1]   ** 2 + detectedPointIrBackL[2]   ** 2)
                if detectionStateIrBackL   else False,
                np.sqrt(detectedPointIrFrontRR[0] ** 2 + detectedPointIrFrontRR[1] ** 2 + detectedPointIrFrontRR[2] ** 2)
                if detectionStateIrFrontRR else False,
                np.sqrt(detectedPointIrFrontR[0] ** 2  + detectedPointIrFrontR[1]  ** 2 + detectedPointIrFrontR[2]  ** 2)
                if detectionStateIrFrontR  else False,
                np.sqrt(detectedPointIrFrontC[0] ** 2  + detectedPointIrFrontC[1]  ** 2 + detectedPointIrFrontC[2]  ** 2)
                if detectionStateIrFrontC   else False,
                np.sqrt(detectedPointIrFrontL[0] ** 2  + detectedPointIrFrontL[1]  ** 2 + detectedPointIrFrontL[2]  ** 2)
                if detectionStateIrFrontL  else False,
                np.sqrt(detectedPointIrFrontLL[0] ** 2 + detectedPointIrFrontLL[1] ** 2 + detectedPointIrFrontLL[2] ** 2)
                if detectionStateIrFrontLL else False]

        # old_min = 0
        # old_max = 0.20
        # new_min = 18000
        # new_max = 0
        # return [(((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min for old_value in vect]
        return vect

    def get_image_front(self):
        return self._get_image(self._FrontalCamera)

    def _get_image(self, camera):
        self.wait_for_ping()

        # get image
        resolution, image = self._vrep_get_vision_sensor_image(camera)

        # reshape image
        image = image[::-1]
        im_cv2 = np.array(image, dtype=np.uint8)
        im_cv2.resize([resolution[0], resolution[1], 3])
        im_cv2 = cv2.flip(im_cv2, 1)

        return im_cv2

    def set_phone_pan(self, pan_position, pan_speed):
        """
        Command the robot to move the smartphone holder in the horizontal (pan) axis.

        Arguments

        pan_position: Angle to position the pan at.
        pan_speed: Movement speed for the pan mechanism.
        """
        # tilt_position = np.pi / 4.0
        self._vrep_set_joint_target_position(self._PanMotor, pan_position)
        self.wait_for_ping()

    def set_phone_tilt(self, tilt_position, tilt_speed):
        """
        Command the robot to move the smartphone holder in the vertical (tilt) axis.

        Arguments

        tilt_position: Angle to position the tilt at.
        tilt_speed: Movement speed for the tilt mechanism.
        """
        # tilt_position = np.pi / 4.0
        self._vrep_set_joint_target_position(self._TiltMotor, tilt_position)
        self.wait_for_ping()
    
    def pause_simulation(self):
        vrep.unwrap_vrep(
            vrep.simxPauseSimulation(self._clientID, vrep.simx_opmode_blocking)
        )
    
    def play_simulation(self):
        vrep.unwrap_vrep(
            vrep.simxStartSimulation(self._clientID, vrep.simx_opmode_blocking)
        )
        self.wait_for_ping()

    def stop_world(self):
        vrep.unwrap_vrep(
            vrep.simxStopSimulation(self._clientID, vrep.simx_opmode_blocking)
        )
        self.wait_for_ping()

    def check_simulation_state(self):
        self.wait_for_ping()
        return vrep.unwrap_vrep(
            vrep.simxGetInMessageInfo(self._clientID, vrep.simx_headeroffset_server_state),
            ignore_novalue_error=True
        )

    def is_simulation_stopped(self):
        return not self.is_simulation_running()

    def is_simulation_running(self):
        info = self.check_simulation_state()
        return info & 1

    def wait_for_stop(self):
        """
        This function busy waits until the simulation is stopped
        """
        while self.is_simulation_running():
            pass

    def get_sim_time(self):
        """
        Gets the simulation time. Returns zero if the simulation is stopped.
        :return: simulation time in milliseconds.
        """
        self.wait_for_ping()
        return vrep.simxGetLastCmdTime(self._clientID)

    def position(self):
        return vrep.unwrap_vrep(
            vrep.simxGetObjectPosition(self._clientID, self._Robobo, -1, vrep.simx_opmode_blocking)
        )

    def collected_food(self):
        ints, floats, strings, buffer = vrep.unwrap_vrep(
            vrep.simxCallScriptFunction(self._clientID, "Food", vrep.sim_scripttype_childscript, "remote_get_collected_food",
                                        [],[],[],bytearray(),vrep.simx_opmode_blocking)
        )
        return ints[0]

    # ----------------

    def reset_scene(self):
        path_robot = 'C:/Program Files/V-REP3/V-REP_PRO_EDU/models/robots/mobile/Robobo_Model.ttm'
        path_scene = 'C:/Program Files/V-REP3/V-REP_PRO_EDU/scenes/OBS_arena.ttt'

        vrep.unwrap_vrep(
            vrep.simxCloseScene(self._clientID, vrep.simx_opmode_blocking)
        )
        
        self.wait_for_ping()

        vrep.unwrap_vrep(
            vrep.simxLoadScene(self._clientID, path_scene, False, vrep.simx_opmode_blocking)
        )
        self.wait_for_ping()
        # vre.unwrapsimxLoadModel(clientID, modelPathAndName, options, operationMode)

    def reset_robot(self, new_position=[0.0, 0.0, 0.0]):
        components =    [self._RightMotor, self._LeftMotor, self._IrBackC, self._IrFrontC, self._IrFrontLL, self._IrFrontRR, self._IrBackL, 
                        self._IrBackLFloor, self._IrBackR, self._IrBackRFloor, self._IrFrontL, self._IrFrontLFloor, self._IrFrontR, self._IrFrontRFloor,
                        self._TiltMotor, self._PanMotor, self._FrontalCamera]

        vrep.simxSetObjectPosition(self._clientID, self._Robobo, -1, new_position, vrep.simx_opmode_blocking)

        for comp in components:
            vrep.unwrap_vrep(
                vrep.simxSetObjectPosition(self._clientID, comp, self._Robobo, new_position, vrep.simx_opmode_blocking)
            )

    def randomize_arena(self, x_rng=(-1.8, 0.025), y_rng=(-0.125, 1.725), z=0.25, safe_space=1.5, n_objects=2):
            # Center of the arena to place robot into
            center = self.position()[0:1]

            placed_objects = []
            obj_names = ['ConcretBlock' + str(i) for i in range(12)]         

            # Remove all objects from the arena
            for object in obj_names:
                handle = self._vrep_get_object_handle(object, vrep.simx_opmode_blocking)

                # randomize position
                vrep.unwrap_vrep(
                    vrep.simxSetObjectPosition(self._clientID, handle, -1, [-5, 0, z],
                                            vrep.simx_opmode_oneshot)
                )             

            for object in obj_names[0:n_objects]:
                valid_pos = False
                while not valid_pos:
                    x = np.random.uniform(*x_rng)
                    y = np.random.uniform(*y_rng)
                    
                    if np.linalg.norm(np.array([x, y]) - center) > safe_space:
                        tqdm.write(f'\tAccepted Dist to robot:{  np.linalg.norm(np.array([x, y]) - center)}')
                        proximity = False
                        for placed_obj in placed_objects:
                            tqdm.write(f'\tdist to other pillar :{ np.linalg.norm(np.array([x, y]) - placed_obj)}')
                            if np.linalg.norm(np.array([x, y]) - placed_obj) < 0.8:
                                proximity = True
                        if not proximity:
                            valid_pos = True

                r = np.random.uniform(-np.pi, np.pi)

                # Get handle from V-REP
                handle = self._vrep_get_object_handle(object, vrep.simx_opmode_blocking)

                # randomize position
                vrep.unwrap_vrep(
                    vrep.simxSetObjectPosition(self._clientID, handle, -1, [x, y, z],
                                            vrep.simx_opmode_oneshot)
                )
                # randomize orientation
                vrep.unwrap_vrep(
                    vrep.simxSetObjectOrientation(self._clientID, handle, -1, [0, 0, r],
                                                vrep.simx_opmode_oneshot)
                )
                self.wait_for_ping()
                placed_objects.append(np.asarray([x, y]))

    def randomize_food_margin(self, x_rng=(-4, -2.25), y_rng=(-0.075, 1.675), z=0.05, safe_space=0.8, n_objects=8):
            # Center of the arena to place robot into
            center = self.position()[0:2]

            placed_objects = []
            obj_names = ['Food' + str(i) for i in range(8)]   

            # Remove all objects from the arena
            for object in obj_names:
                handle = self._vrep_get_object_handle(object, vrep.simx_opmode_blocking)

                # randomize position
                vrep.unwrap_vrep(
                    vrep.simxSetObjectPosition(self._clientID, handle, -1, [-5, 0, z],
                                            vrep.simx_opmode_oneshot)
                )        

            self.wait_for_ping()     
            
            for object in obj_names[0:n_objects]:
                valid_pos = False
                counter = 0
                while not valid_pos:
                    if counter > 25:
                        return placed_objects
                    x = np.random.uniform(*x_rng)
                    y = np.random.uniform(*y_rng)
                    if np.linalg.norm(np.array([x, y]) - np.array(center)) > safe_space:
                        valid_pos = True
                        for placed_obj in placed_objects:
                            if np.linalg.norm(np.array([x, y]) - placed_obj) < (safe_space * 0.75):
                                valid_pos = False
                                break
                    counter += 1

                r = np.random.uniform(-np.pi, np.pi)

                # Get handle from V-REP
                handle = self._vrep_get_object_handle(object, vrep.simx_opmode_blocking)

                # randomize position
                vrep.unwrap_vrep(
                    vrep.simxSetObjectPosition(self._clientID, handle, -1, [x, y, z],
                                            vrep.simx_opmode_oneshot)
                )
                # randomize orientation
                vrep.unwrap_vrep(
                    vrep.simxSetObjectOrientation(self._clientID, handle, -1, [0, 0, r],
                                                vrep.simx_opmode_oneshot)
                )
                self.wait_for_ping()
                placed_objects.append(np.asarray([x, y]))
                
    def toggle_visualization(self):
        vrep.unwrap_vrep(
            vrep.simxSetBooleanParameter(self._clientID, 16, False, vrep.simx_opmode_oneshot)
        )

    def locate_foods(self):
        # Reset locations
        self._foods = []

        for i in range(8):
            # Identify food by name
            name = 'Food{}'.format(i)

            # Determine z-position
            handle = self._vrep_get_object_handle(name, vrep.simx_opmode_blocking)
            x, y, _ = vrep.unwrap_vrep(
                vrep.simxGetObjectPosition(self._clientID, handle, -1, vrep.simx_opmode_blocking)
            )

            # Store handles of the food for fast querying later
            self._foods.append([np.array([x, y]), handle, 'on_ground'])

    def randomize_food(self, x_rng=(-4, -2.25), y_rng=(-0.075, 1.675), safe_space=0.3):
        # Center of the arena to place robot into
        center = np.array([np.mean(x_rng), np.mean(y_rng)])

        foods = ['Food{}'.format(i) for i in range(8)]

        # Reset locations
        self._foods = []

        for name in foods:
            # Place object somewhere within the arena (but not the center)
            x, y = center
            while np.linalg.norm(np.array([x, y]) - center) < safe_space:
                x = np.random.uniform(*x_rng)
                y = np.random.uniform(*y_rng)

            # Determine z-position
            handle = self._vrep_get_object_handle(name, vrep.simx_opmode_blocking)
            _, _, z = vrep.unwrap_vrep(
                vrep.simxGetObjectPosition(self._clientID, handle, -1, vrep.simx_opmode_blocking)
            )

            # Randomize position of Food1 in OBS_Task2.ttt
            vrep.unwrap_vrep(
                vrep.simxSetObjectPosition(self._clientID, handle, -1, [x, y, z], vrep.simx_opmode_oneshot)
            )

            # Store handles of the food for fast querying later
            self._foods.append([np.array([x, y]), handle, 'on_ground'])

    def found_food(self, d_max):
        # Get position of Robobo
        robobo_xy = np.array(self.position())[:2]

        # Check Robobo's current position against all food in arena
        for i, (food_xy, handle, where) in enumerate(self._foods):

            # Check if food is still on the ground and within reach
            if where == 'on_ground' and np.linalg.norm(food_xy - robobo_xy) < d_max:

                # Shoot food into the sky if collided with and set inactive
                self._foods[i][2] = 'poof'
                vrep.unwrap_vrep(
                    vrep.simxSetObjectPosition(self._clientID, handle, -1, list(food_xy) + [10], vrep.simx_opmode_oneshot)
                )

                return True

        return False
