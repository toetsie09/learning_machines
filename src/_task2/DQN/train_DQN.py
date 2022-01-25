import tqdm
import numpy as np
import torch
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_interface import RoboboEnv
from DQN import DQNAgent

def compute_reward(collision):
    '''
        Collision should be an integer: 
            0 for no collision, 
            1 for collision with wall, 
            2 for collision with food
    '''
    reward = -1
    if collision == 1:
        reward = -100
    elif collision == 2:
        reward = 100
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
        # pbar = tqdm(total=n_steps, position=0, desc=str(i_episode), leave=True)
        rewards = []

        robot.start()
        for i_step in range(n_steps):
            state = robot.get_sensor_state()
            state = ir_to_proximity(state)

            action = controller.select_action(i_episode, n_episodes, state)
            robot.take_action(action)

            next_state = robot.get_sensor_state()
            collision = robot.has_collided()

            collision = 0 if collision == False else 1 # Temporary
            
            reward = compute_reward(collision)
            rewards.append(reward)  

            print(f'Take action {action}, collision: {collision} and reward: {reward}')
            
            controller._memory.push(torch.tensor(state), action, torch.tensor(next_state), reward)
            controller.optimize_model()
            print('Updated the networks')

            if collision == 1:
                break

            # pbar.set_postfix({'reward': reward})
            # pbar.update(1)

        robot.stop()
        # pbar.set_postfix({'avg_reward': np.mean(rewards)})
        # pbar.close()

        if i_episode % 2 == 0:
            controller._target_network.load_state_dict(controller._policy_network.state_dict())
            print('Overwriting the target network')
            controller.save_models('./src/models/', name='test')

if __name__ == "__main__":
    # Initialize robobo
    robobo = RoboboEnv(env_type='simulation', robot_id='#0', ip='192.168.192.14', hide_render=False, camera=True)  # 192.168.192.14 - Sander

    # Initialize controller
    DQN_controller = DQNAgent(n_inputs=5, n_hidden=24, n_outputs=4, gamma=0.5)

    # Train controller
    train_controller(robobo, DQN_controller, n_episodes=200, n_steps=50)