import gymnasium as gym
from stable_baselines3 import PPO

# Criar um ambiente de teste
env = gym.make("CartPole-v1")

# Criar um modelo PPO e treinar rapidamente
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

print("Teste concluído com sucesso!")
