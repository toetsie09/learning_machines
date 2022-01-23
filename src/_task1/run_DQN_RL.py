import torch
import numpy as np

from DQN import DQN, Robobo_Controller

def load_model(PATH, input_size, num_actions, DEVICE):
    model = DQN(input_size, num_actions, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

def main(n_episodes=10):
    controller = Robobo_Controller(RL=True, address='192.168.43.248')

    DEVICE = torch.device('cpu')
    N_ACTIONS = 3
    N_INPUTS = 5

    PATH = './src/models/'

    policy_net = load_model(PATH + 'DQN_policy_v2.pt', N_INPUTS, N_ACTIONS, DEVICE)
    # policy_net = load_model(PATH + 'DQN_target_v1.pt', N_INPUTS, N_ACTIONS, DEVICE)

    n_collisions = 0

    current_state = controller.get_state(True)

    for i in range(n_episodes):
        # print('state:', current_state)
        next_action = policy_net.forward(current_state).argmax().item()
        print(next_action, next_action.argmax().item())
        controller.take_action_RL(next_action)
        current_state = controller.get_state(False)
        # print('state:', current_state)
        collision, front_bool, back_bool = controller.detect_collision_RL(current_state)
        # print('collision', collision)
        
        if collision == 3 and not front_bool:
            n_collisions += 1

            reverse_counter = 0
            print('Robot collided, fixing this now\n')
            while not front_bool:
                if reverse_counter > 2:
                    break
                controller.rob.move(-10, -5, 2000)
                state = controller.get_state(False)
                _, front_bool, _ = controller.detect_collision_RL(state)
                reverse_counter += 1

        current_state = torch.Tensor(np.asarray(current_state))
        
if __name__ == "__main__":
    # environment = environment_setup()
    main(100)