import gymnasium as gym
import robo_gym
from PIL import Image as PILImage
import os

target_machine_ip = "192.168.1.23:50051"
robot_model = "bunker"
env = gym.make(
    "BunkerRRob-v0",
    rs_address=target_machine_ip,
    robot_model=robot_model,
    with_camera=True,
)

action = [0.1, 0]
index_img = 0
for episode in range(1):
    done = False
    env.reset()
    while not done:
        state, reward, done, _, info = env.step(action)
        print(state["camera"].shape)
        debug_img = PILImage.fromarray(state['camera'])
        debug_img_dir = f"./debug/"
        if not os.path.exists(debug_img_dir):
            os.makedirs(debug_img_dir)

        debug_img.save(os.path.join(debug_img_dir, f"img_{index_img}.png"))
        index_img += 1
