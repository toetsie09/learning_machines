### TASK 2

import numpy as np
import torch
import cv2

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

def test_controller(robot, controller, controller_path):
    controller.load_models(controller_path)

    while True:
        image = robot.take_picture()
        state = torch.tensor(identify_food(image))
        action = controller._policy_network.forward(state).argmax().item()
        robot.take_action(action, SIM=False)

if __name__ == "__main__":
    # Initialize robobo
    robobo = RoboboEnv(env_type='hardware', robot_id='', camera=True)

    # Initialize controller
    DQN_controller = DQNAgent(n_inputs=3, n_hidden=24, n_outputs=4)

    controller_path = './src/_task2/DQN/models/DQN_policy_network_test.pt'

    # Train controller
    test_controller(robobo, DQN_controller, controller_path)