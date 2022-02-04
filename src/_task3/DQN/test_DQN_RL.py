import numpy as np
import torch
import cv2

from robot_interface_DQN import RoboboEnv
from DQN import DQNAgent

# Green lair
BASE_HSV_MIN = (36, 50, 70)
BASE_HSV_MAX = (96, 240, 240)

# Red food (two sides of hsv)
FOOD_HSV_MIN = [(160, 50, 70), (0, 50, 70)]
FOOD_HSV_MAX = [(180, 255, 240), (10, 255, 240)]

def identify_object(img, min_hsv, max_hsv, min_blob_size=64):
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

        # Processing!
    img = img[::-1, ::-1]
    img = cv2.resize(img, (256, 256))
    # img = cv2.medianBlur(img, 3)

    food = identify_object(img, FOOD_HSV_MIN, FOOD_HSV_MAX, min_blob_size=8)
    base = identify_object(img, BASE_HSV_MIN, BASE_HSV_MAX, min_blob_size=8)

    # Check if some object is in gripper
    # obj_in_gripper = in_gripper(robot.get_sensor_state())

    # return np.concatenate([food, base, obj_in_gripper], axis=0)
    return np.concatenate([food, base], axis=0)

def test_controller(robot, controller, controller_path):
    controller.load_models(controller_path)

    while True:
        state = get_state(robot) 
        print(state)

        action = controller._policy_network.forward(torch.tensor(state)).argmax().item()
        robot.take_action(action, SIM=False)    

if __name__ == "__main__":
    # Initialize robobo
    robobo = RoboboEnv(env_type='hardware', robot_id='', ip='192.168.46.237', camera=True)  # 192.168.192.14 - Sander
    print('robot initalized')

    # Initialize controller
    DQN_controller = DQNAgent(n_inputs=6, n_hidden=24, n_outputs=4)

    controller_path = './src/_task3/DQN/models/DQN_policy_network_stuckreset.pt'
    # Train controller
    test_controller(robobo, DQN_controller, controller_path)