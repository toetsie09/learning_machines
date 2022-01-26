from multiprocessing.dummy import current_process
from tqdm import tqdm
import numpy as np
import torch
import sys, os
import cv2

from robot_interface import RoboboEnv
from DQN import DQNAgent

FOOD_HSV_MIN = (36, 0, 0)
FOOD_HSV_MAX = (86, 255, 255)

def identify_food(img, min_hsv=FOOD_HSV_MIN, max_hsv=FOOD_HSV_MAX):
    # Convert to Hue-Saturation-Value (HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask out parts of original image that lie in color range
    mask = cv2.inRange(img, min_hsv, max_hsv)

    # Try to find blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return np.array([1, 0, 0])  # ('no food', x, y)

    # Select largest contour from the mask
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 8:
        return np.array([1, 0, 0])  # ('no food', x, y)

    # Determine center of the blob
    M = cv2.moments(largest_contour)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # Normalize x and y relative to image centroid and image width/height
    y_norm = 2 * (y / img.shape[0]) - 1
    x_norm = 2 * (x / img.shape[1]) - 1
    return np.array([0, x_norm, y_norm])

def compute_reward(collision, food_collected):
    if food_collected > 0:
        reward = 100
    elif collision:
        reward = -100
    else:
        reward = -1
    return torch.tensor([reward])

def ir_to_proximity(sim_sensor_dists, d_max=0.2):
    """ Convert distances to proximity values
    """
    values = []
    for d in sim_sensor_dists:
        if type(d) == bool:
            values += [0]
        else:
            values += [max(0, (d_max - d) / d_max)]
    return np.array(values)

def train_controller(robot, controller, n_episodes=200, n_steps=50):
    for i_episode in range(n_episodes):
        pbar = tqdm(total=n_steps, position=0, desc=f'Current episode: {i_episode}', leave=True)
        rewards = []
        
        food_locations = robot.start(safe_space=0.5, n_objects=5)
        # print('arena started and randomized')
        for i_step in range(n_steps):
            start_collected_food = robot._env.collected_food()

            image = robot.get_front_image()
            state = identify_food(image)

            action = controller.select_action(i_episode, n_episodes, state)
            robot.take_action(action)

            image = robot.get_front_image()
            next_state = identify_food(image)
            end_collected_food = robot._env.collected_food()
            food_collected = end_collected_food - start_collected_food

            collision = robot.has_collided()
            
            reward = compute_reward(collision, food_collected)
            rewards.append(reward.item())  

            if food_collected > 0:
                print(f'\nCollected food, reward:{reward}')
            if collision:
                print(f'\nCollision, reward: {reward}')


            # print(f'Take action {action}, collision: {collision} and reward: {reward}')
            
            controller._memory.push(torch.tensor(state), action, torch.tensor(next_state), reward)
            controller.optimize_model()
            # print('Updated the networks')

            if collision == 1:
                break

            pbar.set_postfix({'reward': reward.item()})
            pbar.update(1)

        robot.stop()
        pbar.set_postfix({'avg_reward': np.mean(rewards)})
        pbar.close()

        if i_episode % 10 == 0:
            controller._target_network.load_state_dict(controller._policy_network.state_dict())
            # print('Overwriting the target network')
            controller.save_models('./src/models/', name='test')

if __name__ == "__main__":
    # Initialize robobo
    robobo = RoboboEnv(env_type='randomized_simulation', robot_id='', ip='192.168.192.14', hide_render=False, camera=True)  # 192.168.192.14 - Sander
    print('robot initalized')

    # Initialize controller
    DQN_controller = DQNAgent(n_inputs=3, n_hidden=24, n_outputs=4, gamma=0.6)

    # Train controller
    train_controller(robobo, DQN_controller, n_episodes=200, n_steps=500)