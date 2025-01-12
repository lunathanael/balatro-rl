from main import Agent
import gymnasium as gym
import balatro_gym as bg
import torch
import numpy as np

env = gym.make("Balatro-v0", render_mode="text")

path = "models/Balatro-v0__punish__1925__1736660706.pth"
agent = Agent.load(path, np.array(env.observation_space.shape).prod(), env.action_space.n)

obs, info = env.reset()
env.render()
action = None
for i in range(100):
    obs = torch.Tensor(obs)
    action = agent.get_action_and_value(obs)
    # env.render()
    # print(action)
    obs, reward, done, truncated, info = env.step(action[0].numpy())
    if action[0] >= 52:
        print(reward, int(action[-1]))
        env.render()
    # input()
    if done or truncated:
        break

env.close()
