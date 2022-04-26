from the_beer_game_environment import BeerGameEnv


env = BeerGameEnv()
episodes = 10

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		print("action", random_action)
		obs, reward, done, info = env.step(random_action)
		print('reward', reward)