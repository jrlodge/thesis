from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import os
from the_beer_game_environment import BeerGameEnv
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

TIMESTEPS = 1000

writer = SummaryWriter('the_beer_game/ddpg')

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = BeerGameEnv()
env.reset()

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env=env, action_noise=action_noise, tensorboard_log=logdir, verbose=1)
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=10, tb_log_name=f"DDPG_LOG")

'''iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=10, tb_log_name=f"DDPG")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")'''