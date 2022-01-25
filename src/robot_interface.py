from enum import Enum
import cv2
from matplotlib import pyplot as plt
from tqdm import trange
import robobo
import numpy as np
import socket

class Emote(Enum):
    #emotions found in: https://education.theroboboproject.com/en/scratch3/smartphone-actuation-blocks
    HAPPY= "happy"
    LAUGHING= "laughing"
    SURPRISED= "surprised"
    SAD= "sad"
    ANGRY= "angry"
    NORMAL= "normal"
    SLEEPING= "sleeping"
    TIRED= "tired"
    AFRAID= "afraid"

class RoboboEnv:
    def __init__(self, env_type='simulation', ip='192.168.1.113', robot_id='#0', used_sensors='5', hide_render=True):
        # Init simulated or hardware arena
        if env_type == 'simulation' or env_type == 'randomized_simulation':
            self._env = robobo.SimulationRobobo(robot_id).connect(address=socket.gethostbyname(socket.gethostname()), port=19997)
        elif env_type == 'hardware':
            self._env = robobo.HardwareRobobo(camera=True).connect(address=ip)
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
            #if self._hide_render:
            #    self._env.toggle_visualization()

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

    def countDown(self, seconds=1):
        while seconds > 0:
            self.sleep(1)
            self._env.set_emotion(Emote.SURPRISED.value if seconds -1 == 0 else Emote.LAUGHING.value)
            self._env.talk(f"{seconds}")
            seconds -= 1
        return self._env.set_emotion(Emote.NORMAL.value)

    def PlotColorRanges(self, image, filename, HSV=True):
        color = ('b','g','r')
        for j,col in enumerate(color):
            histr = cv2.calcHist([image],[j],None,[256],[0,256])
            plt.plot(histr,color = col, label=col)
            plt.xlim([0,256])

        plt.legend()
        plt.savefig(filename+"_RGB.png")
        plt.clf()

        if HSV:
            imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = ('c','m','y')
            for j,col in enumerate(hsv):
                histr = cv2.calcHist([imageHSV],[j],None,[256],[0,256])
                plt.plot(histr,color = col, label="hsv"[j])
                plt.xlim([0,256])

            plt.legend()
            plt.savefig(filename+"_HSV.png")
            plt.clf()
            #plt.show()
        return

    def get_front_image(self):
        image = self._env.get_image_front()
        image = image[160:, :] #crop lowersection HxW 640x480 -> 480x480
        image = cv2.medianBlur(image,ksize=5) # smooth image to denoise it
        return image 

    def takePhoto(self ,times=1, count=3):
        for i in trange(times):
            self.countDown(count)

            self._env.talk("Smile!")
            image = self.get_front_image()

            self.PlotColorRanges(image, f"./src/view/food/color_hist_{i}")

            if  self.in_simulation:
                cv2.destroyAllWindows()
                cv2.imshow("Live view",image)
            else:
                cv2.imwrite(f"./src/view/food/test_photo_{i}.png",image)
        self._env.talk("Finished taking photos.")
        return
    
    @staticmethod
    def BlobDetect(frame_threshed, minThreshold= 0, maxThreshold= 255, 
                minArea= 1500, maxArea=230_400, minCircularity=0.0, maxCircularity= 0.9,
                minConvexity= 0.87, maxConvexity= 1,
                minInertiaRatio= 0.01,maxInertiaRatio= 1):

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = minThreshold
        params.maxThreshold = maxThreshold

        # Filter by Area.
        params.filterByArea = True
        params.minArea = minArea
        params.maxArea = maxArea #480*480= 230_400

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = minCircularity
        params.maxCircularity = maxCircularity

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = minConvexity
        params.maxConvexity = maxConvexity

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = minInertiaRatio
        params.maxInertiaRatio = maxInertiaRatio

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 : # need version 3 or 4
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(frame_threshed)
        #print(np.shape(keypoints))
        #print(keypoints)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        image_with_keypoints = cv2.drawKeypoints(frame_threshed, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        #TODO:
        #convert the keypoints into inputs usable for the Neural net
        
        return keypoints, image_with_keypoints