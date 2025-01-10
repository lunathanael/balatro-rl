from main import Agent
import gymnasium as gym
import balatro_gym as bg
import torch

env = gym.make("Balatro-v0", render_mode="text")

agent = Agent.load("models/Balatro-v0__main__1__1736524763_4997120.pth")

obs, info = env.reset()

for i in range(100):
    obs = torch.Tensor(obs)
    action = agent.get_action_and_value(obs)
    _action = agent.actor(agent.network(obs))
    env.render()
    print(action)
    print(obs.reshape(4, 13),_action.detach().numpy().reshape(4, 13))
    obs, reward, done, truncated, info = env.step(action[0].numpy())
    print(reward)
    input()
    if done or truncated:
        break
print(action, reward)
env.close()
