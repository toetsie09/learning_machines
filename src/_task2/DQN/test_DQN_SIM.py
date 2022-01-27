import numpy as np
import torch
import cv2
from tqdm import tqdm
import pickle

from robot_interface_DQN import RoboboEnv
from DQN import DQNAgent

FOOD_HSV_MIN = (36, 70, 50)
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

def test_controller(robot, controller, n_episodes=50, n_steps=200):
    controller.load_models('./src/_task2/DQN/models/', 'DQN_policy_network_test.pt')

    overall_rewards = []
    overall_food = []
    overall_survive_duration = []

    max_food = 7
    for i_episode in range(n_episodes):
        pbar = tqdm(total=n_steps, position=0, desc=f'Current episode: {i_episode}', leave=True)

        rewards_per_episode = []

        robot.start(0.5, max_food)

        for i_step in range(n_steps):
            start_collected_food = robot._env.collected_food()
            image = robot.take_picture()
            state = identify_food(image)
            state = torch.tensor(state)
            
            action = controller._policy_network.forward(state).argmax().item()
            robot.take_action(action)
            end_collected_food = robot._env.collected_food()

            food_collected = end_collected_food - start_collected_food
            collision = robot.has_collided()
            
            reward = compute_reward(collision, food_collected)

            rewards_per_episode.append(reward.item())  

            pbar.set_postfix({'reward': reward.item()})
            pbar.update(1)

            if collision:
                overall_survive_duration.append(i_step)
                break

            if robot._env.collected_food() >= max_food:
                break

        if not collision:
            overall_survive_duration.append(n_steps)
        
        overall_rewards.append(rewards_per_episode)
        overall_food.append(robot._env.collected_food())

        robot.stop()
        pbar.set_postfix({'avg_reward': np.mean(rewards_per_episode)})
        print(rewards_per_episode)
        pbar.close()
    
        if i_episode % 10 == 0 or i_episode == n_episodes-1:
            path = './src/_task2/DQN/results/'
            # Save training stats
            with open(path + 'DQN_test_rewards_v1.pkl', 'wb') as file:
                pickle.dump(overall_rewards, file)

            with open(path + 'DQN_test_collected_foods_v1.pkl', 'wb') as file:
                pickle.dump(overall_food, file)

            with open(path + 'DQN_test_survive_duration_v1.pkl', 'wb') as file:
                pickle.dump(overall_survive_duration, file)

if __name__ == "__main__":
    # Initialize robobo
    robobo = RoboboEnv(env_type='randomized_simulation', robot_id='', hide_render=False, camera=True)  # 192.168.192.14 - Sander
    print('robot initalized')

    # Initialize controller
    DQN_controller = DQNAgent(n_inputs=3, n_hidden=24, n_outputs=4)

    # Train controller
    test_controller(robobo, DQN_controller, n_episodes=50, n_steps=300)