from main import Agent
import gymnasium as gym
import balatro_gym as bg
import torch

env = gym.make("Balatro-v0", render_mode="text")

agent = Agent.load("models/Balatro-v0__main__1__1736548772_1843200.pth")

obs, info = env.reset()
action = None
for i in range(100):
    obs = torch.Tensor(obs)
    if action is None or action[0] == 52:
        env.render()
    action = agent.get_action_and_value(obs)
    # env.render()
    # print(action)
    obs, reward, done, truncated, info = env.step(action[0].numpy())
    # print(reward)
    # input()
    if done or truncated:
        break
print(action, reward)
env.close()
