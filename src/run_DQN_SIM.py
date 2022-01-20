import torch
import numpy as np

from DQN import DQN, Robobo_Controller

def load_model(PATH, input_size, num_actions, DEVICE):
    model = DQN(input_size, num_actions, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

def main(n_episodes=10):
    controller = Robobo_Controller()

    DEVICE = torch.device('cpu')
    N_ACTIONS = 3
    N_INPUTS = 5

    PATH = './src/models/'

    policy_net = load_model(PATH + 'DQN_policy_v2.pt', N_INPUTS, N_ACTIONS, DEVICE)
    # policy_net = load_model(PATH + 'DQN_target_v1.pt', N_INPUTS, N_ACTIONS, DEVICE)

    n_collisions = 0
    print(controller.get_position())
    current_state = controller.get_state(True)

    steps_per_env = []

    for i in range(n_episodes):
        controller.reset_arena()
        # n_steps = 0
        # collision = 0

        # while collision != 3 and n_steps <= 100:
        #     next_action = policy_net.forward(current_state).argmax().item()
        #     controller.take_action_RL(next_action)
        #     current_state = controller.get_state(False)
        #     collision, _, _ = controller.detect_collision_SIM(current_state)

        #     current_state = torch.Tensor(np.asarray(current_state))
        #     n_steps += 1

        # steps_per_env.append(n_steps)

    controller.rob.stop_world()
    controller.rob.wait_for_stop()
        
if __name__ == "__main__":
    # environment = environment_setup()
    main(3)