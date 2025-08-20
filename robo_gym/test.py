import gymnasium as gym
import robo_gym


target_machine_ip = '127.0.0.1:50051'
robot_model = 'bunker'
env = gym.make('BunkerRRob-v0', rs_address=target_machine_ip, robot_model=robot_model, with_camera=True)

action = [0.1,0]
for episode in range(1):
    done = False
    env.reset()
    while not done:
        state, reward, done,_, info = env.step(action)
        print(state)