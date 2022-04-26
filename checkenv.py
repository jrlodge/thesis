from stable_baselines3.common.env_checker import check_env
from the_beer_game_environment import BeerGameEnv


env = BeerGameEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)