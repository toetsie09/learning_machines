#!/usr/bin/env python3

#python3.7 w tensorflow1.15!
import time
import numpy as np

import socket
import robobo
import sys
import signal

from gym import Env
from gym.spaces import Box

from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def RobotPosition(robot: robobo.SimulationRobobo, accuracy: int = 10, decimals: int = 3):
    measurements = []
    for _ in range(accuracy):
        measurements.append(robot.position())
    measurements = np.array(measurements)
    #print([np.std(measurements[:, i]) for i in range(3)])
    return [np.round(np.mean(measurements[:, i]), decimals=decimals) for i in range(3)] # np.median


def ResetEnv(robot: robobo.SimulationRobobo):
    robot.stop_world()
    robot.wait_for_stop()

    robot.play_simulation()
    return


class VirtualRoboboEnv(Env):
    def __init__(self):
        # Actions we can take: boost left & right motor speed (-5/5)
        self.action_space = Box(low=np.array([-15,-15]), high=np.array([15,15]))
        # IR-sensor range
        self.observation_space = Box(low=np.array([0.0 for _ in range(8)]), high=np.array([0.3 for _ in range(8)]))

        #episode lenght
        self.EP_LENGHT=100
        self.stepsLeft = self.EP_LENGHT
        
        signal.signal(signal.SIGINT, terminate_program) #TODO
        self.ROBOT:robobo.SimulationRobobo= robobo.SimulationRobobo("#0").connect(
        address=socket.gethostbyname(socket.gethostname()), port=19997)
        self.ROBOT.play_simulation()
        time.sleep(1)
        # Set start state
        self.state = self.ROBOT.read_irs()

        
    def step(self, action):
        # Reduce EP length by 1 step
        self.stepsLeft -= 1 
        
        # Apply action
        MOVETIME= 100
        self.ROBOT.move(action[0]+1, action[1]+1, MOVETIME)
        time.sleep(MOVETIME/1000) #wait till finished moving
        observation = self.ROBOT.read_irs()
        self.state = observation

        # Calculate reward
        if any([i < 0.03 for i in observation]) :#or all([i == False for i in observation]) != True:
            #collision
            print("---collision---")
            self.ROBOT.talk('Oops')
            reward= -100
        else:
            reward=  action[0] +action[1] - abs(action[0] - action[1])
            if action[0] == action[1] == 5:
                reward+= 10
         
        # Check if EP is done
        if self.stepsLeft <= 0: 
            done = True
            #self.end()
        else:
            done = False
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Already done by VREP!
        # dont remove plz
        pass
    
    def reset(self):
        ResetEnv(self.ROBOT)
        # Reset state
        self.state =  self.ROBOT.read_irs() 

        # Reset EP time
        self.stepsLeft = self.EP_LENGHT 
        return self.state
    
    def end(self):
        # pause the simulation and read the collected food
        self.ROBOT.set_emotion('sad')
        self.ROBOT.talk('I am stopping. Goodbye')
        #self.ROBOT.pause_simulation()

        # Stopping the simualtion resets the environment
        self.ROBOT.stop_world()
        self.ROBOT.wait_for_stop()
        pass


def main():
    env = VirtualRoboboEnv()

    #used this lib->https://stable-baselines.readthedocs.io/en/master/modules/td3.html
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=5000, log_interval=10)
    model.save("models/td3.model")

    '''episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            #env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))'''



if __name__ == "__main__":
    main()
