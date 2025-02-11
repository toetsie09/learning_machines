### TASK 1
from tqdm import tqdm
import numpy as np
import torch
import pickle
import cv2

from robot_interface_DQN import RoboboEnv
from DQN import DQNAgent

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

def compute_reward(collision, distance, action, front_bool):
    """
        front_bool: true if the front of the robot is clear of objects
    """
    if collision:
        return torch.tensor([-100])

    multiplyer = 1
    if front_bool: # Penalize moving backwards and reward moving straight when the front is clear
        multiplyer = 1.2 if action.item() == 1 else -0.2 if action.item() == 4 else 1

def check_front(sensor_dists, d_min=0.08):
    """
        This function return True if the front of the robot is clear
    """
    for s in sensor_dists[2:]:
        if s > d_min:
            return False
    return True

def test_controller(robot, controller, n_episodes, n_steps, controller_name, result_path, experiment_name, max_objects=12, object_distance=0.3):
    controller.load_models(controller_name)
    overall_rewards = []
    distance_travelled = []

    for i_episode in range(n_episodes):
        pbar = tqdm(total=n_steps, position=0, desc=f'Current episode: {i_episode}', leave=True)
        rewards_per_episode = []
        positions = []

        robot.start(object_distance, max_objects)
        for i_step in range(n_steps):
            state = ir_to_proximity(robot.get_sensor_state())
            front_bool = check_front(state)

            start_position = np.array(robot.position)

            action = controller._policy_network.forward(torch.tensor(state)).argmax()
            robot.take_action(action)

            end_position = np.array(robot.position)
            distance = np.linalg.norm(start_position - end_position)
            collision = robot.has_collided()

            reward = compute_reward(collision, distance, action, front_bool)
            rewards_per_episode.append(reward.item())

            pbar.set_postfix({'reward': reward.item()})
            pbar.update(1)

            positions.append(robot.position)

            if collision:
                break

        overall_rewards.append(rewards_per_episode)

        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        distance_travelled.append(path_length)

        robot.stop()
        pbar.set_postfix({'avg_reward': np.mean(rewards_per_episode)})
        pbar.close()

    # Save final testing stats
    with open(f'{result_path}DQN_test_rewards_{experiment_name}.pkl', 'wb') as file:
        pickle.dump(overall_rewards, file)

    with open(f'{result_path}DQN_test_collected_foods_{experiment_name}.pkl', 'wb') as file:
        pickle.dump(distance_travelled, file)
        
if __name__ == "__main__":
    # Initialize robobo
    robobo = RoboboEnv(env_type='randomized_simulation', robot_id='', hide_render=False, task=1)

    # Initialize controller
    DQN_controller = DQNAgent(n_inputs=5, n_hidden=24, n_outputs=4, gamma=0.99)

    # Set variables for training
    result_path = './src/_task1/DQN/results/'
    controller_name = './src/_task1/DQN/models/DQN_policy_network_prefer_forwards.pt'
    experiment_name = 'prefer_forwards'
    n_episodes = 300
    n_steps = 500

    # Train controller
    test_controller(robobo, DQN_controller, n_episodes, n_steps, controller_name, result_path, experiment_name)