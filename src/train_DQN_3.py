### TASK 2

from tqdm import tqdm
import numpy as np
import torch
import pickle
import cv2

from robot_interface_DQN import RoboboEnv
from DQN import DQNAgent

# Green lair
BASE_HSV_MIN = (36, 50, 70)
BASE_HSV_MAX = (86, 255, 255)

# Red food (two sides of hsv)
FOOD_HSV_MIN = [(160, 50, 70), (0, 50, 70)]
FOOD_HSV_MAX = [(180, 255, 255), (10, 255, 255)]

def identify_object(img, min_hsv, max_hsv, min_blob_size=8):
    # Convert to Hue-Saturation-Value (HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask out parts of original image that lie in color range
    if type(min_hsv) == tuple:
        mask = cv2.inRange(img, min_hsv, max_hsv)
    else:
        mask0 = cv2.inRange(img, min_hsv[0], max_hsv[0])
        mask1 = cv2.inRange(img, min_hsv[1], max_hsv[1])
        mask = cv2.bitwise_or(mask0, mask1)

    # Try to find blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return np.array([1, 0, 0])  # ('not found', x, y)

    # Select largest contour from the mask
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_blob_size:
        return np.array([1, 0, 0])  # ('not found', x, y)

    # Determine center of the blob
    M = cv2.moments(largest_contour)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # Normalize x and y relative to image centroid and image width/height
    y_norm = 2 * (y / img.shape[0]) - 1
    x_norm = 2 * (x / img.shape[1]) - 1
    return np.array([0, x_norm, y_norm])  # ('found', x, y)

def in_gripper(ir_sensors, thres=0.1):
    value = ir_sensors[3]  # Front-C in reduced sensor array
    if type(value) != bool and value < thres:
        return True
    return False

def compute_reward(in_gripper, move_to_food, move_to_base):
    multiplyer = 10 if in_gripper and move_to_base >= 0 else 2
    reward = move_to_food + multiplyer * move_to_base
    return torch.tensor([reward])

def get_state(robot):
    # Locate Food and Lair objects in camera image
    img = robot.take_picture()
    food = identify_object(img, FOOD_HSV_MIN, FOOD_HSV_MAX, min_blob_size=8)
    base = identify_object(img, BASE_HSV_MIN, BASE_HSV_MAX, min_blob_size=8)

    # Check if some object is in gripper
    # obj_in_gripper = in_gripper(robot.get_sensor_state())

    # return np.concatenate([food, base, obj_in_gripper], axis=0)
    return np.concatenate([food, base], axis=0)

def train_controller(robot, controller, n_episodes, n_steps, result_path, model_path, experiment_name, object_distance=0.5):
    overall_rewards = []
    distance_food_base = []
    distance_robot_food = []

    for i_episode in range(n_episodes):
        pbar = tqdm(total=n_steps, position=0, desc=f'Current episode: {i_episode}', leave=True)
        rewards_per_episode = []
        stuck_counter = 0

        robot.start(object_distance)
        
        # Observe state before action
        state = get_state(robot)
        dist_robot_to_food = robot.distance_between('Robobo', 'Food')
        dist_food_to_base = robot.distance_between('Food', 'Base')  

        for i_step in range(n_steps):
            start_position = np.array(robot.position)

            # Select and execute action
            action = controller.select_action(i_episode, n_episodes, state)
            robot.take_action(action)

            # Observe state after action
            next_state = get_state(robot)
            new_dist_robot_to_food = robot.distance_between('Robobo', 'Food')
            new_dist_food_to_base = robot.distance_between('Food', 'Base')

            end_position = np.array(robot.position)
            distance = np.linalg.norm(start_position - end_position)

            # Reward parameters
            food_in_gripper = in_gripper(robot.get_sensor_state())
            move_to_food = dist_robot_to_food - new_dist_robot_to_food
            move_to_base = dist_food_to_base - new_dist_food_to_base
            
            # Compute rewards
            reward = compute_reward(food_in_gripper, move_to_food, move_to_base)
            rewards_per_episode.append(reward.item())  
           
            # Update memory and networks
            controller._memory.push(torch.tensor(state), action, torch.tensor(next_state), reward)
            controller.optimize_model()

            state = next_state
            dist_robot_to_food = new_dist_robot_to_food
            dist_food_to_base = new_dist_food_to_base

            pbar.set_postfix({'reward': reward.item()})
            pbar.update(1)

            if distance < 0.005:
                stuck_counter += 1
            else:
                stuck_counter = 0

            if new_dist_food_to_base < 0.1 or stuck_counter >= 6:
                break

        overall_rewards.append(rewards_per_episode)
        distance_food_base.append(dist_food_to_base)
        distance_robot_food.append(dist_robot_to_food)

        robot.stop()
        pbar.set_postfix({'avg_reward': np.mean(rewards_per_episode)})
        pbar.close()

        if i_episode % 10 == 0: # Update the target network every 10 episodes
            controller._target_network.load_state_dict(controller._policy_network.state_dict())
            controller._target_network.eval()
            
        if i_episode % 30 == 0: # Save intermediate training stats every 30 episodes
            controller.save_models(model_path, name=experiment_name)
            with open(f'{result_path}DQN_training_rewards_{experiment_name}.pkl', 'wb') as file:
                pickle.dump(overall_rewards, file)

            with open(f'{result_path}DQN_training_food_base_{experiment_name}.pkl', 'wb') as file:
                pickle.dump(distance_food_base, file)

            with open(f'{result_path}DQN_training_robot_food_{experiment_name}.pkl', 'wb') as file:
                pickle.dump(distance_robot_food, file)
        
    # Save final training stats
    controller.save_models(model_path, name=experiment_name)

    with open(f'{result_path}DQN_training_rewards_{experiment_name}.pkl', 'wb') as file:
        pickle.dump(overall_rewards, file)

    with open(f'{result_path}DQN_training_food_base_{experiment_name}.pkl', 'wb') as file:
        pickle.dump(distance_food_base, file)

    with open(f'{result_path}DQN_training_robot_food_{experiment_name}.pkl', 'wb') as file:
        pickle.dump(distance_robot_food, file)

if __name__ == "__main__":
    # Initialize robobo
    robobo = RoboboEnv(env_type='randomized_simulation', robot_id='', hide_render=False, task=3)

    # Initialize controller
    DQN_controller = DQNAgent(n_inputs=6, n_hidden=24, n_outputs=4, gamma=0.99)

    # Set variables for training
    result_path = './src/_task3/DQN/results/'
    model_path = './src/_task3/DQN/models/'
    experiment_name = 'stuckreset'
    n_episodes = 300
    n_steps = 200

    # Train controller
    train_controller(robobo, DQN_controller, n_episodes, n_steps, result_path, model_path, experiment_name)