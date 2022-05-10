import gym
import numpy as np
from the_beer_game_environment import BeerGameEnv
from stable_baselines3 import SAC

env = BeerGameEnv()

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=1)
model.save("sac_beer_game")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_beer_game")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()