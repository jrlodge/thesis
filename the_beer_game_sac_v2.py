from the_beer_game_environment import BeerGameEnv
from stable_baselines3 import SAC
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt

num_inputs = 8
num_actions = 100
num_hidden = 64

# I have changes the architecture a bit, because I think this will be more better. 
# I have just allocated one dense layer before actor and critic, that will be specific for 
# each one, and all other network will be common. This will make it more accurate. 
    
inputs = layers.Input(shape=(num_inputs,))
dense = layers.Dense(64, activation="relu")(inputs)
dense = layers.Dense(64, activation="relu")(dense)
dense = layers.Dense(64, activation="relu")(dense)
dense1 = layers.Dense(64, activation="relu")(dense)
dense2 = layers.Dense(64, activation="relu")(dense)
action = layers.Dense(num_actions, activation="softmax")(dense1)
critic = layers.Dense(1)(dense2)

model = keras.Model(inputs=inputs, outputs=[action, critic])

model.build(input_shape = num_inputs)

# We are doing it first time so we will ignore this part for now. 
# Load these weights when we have fully trained network otherwise comment this 

#model.load_weights("Final_Weights_Actor_Critic.h5")


# These are weights that are saved at the end of every episode. 
# In case you need to turn off the system or due to some problem the system turns off during 
# training you will load thses weights so that you can resume last training. 

#model.load_weights("Actor_Critic.h5")

env = BeerGameEnv()

# You can change this further. If you are looking for long term reward that means 
# reward that will be at the end of the episode is more important then keep this term 0.98. (I don't think this is true in your case, but not sure) 
# otherwise, if you think that reward at each step is important then make this term 0.89 or 0.79. <<--You can test both one by one. 
gamma = 0.98 
optimizer = keras.optimizers.Adam(learning_rate=0.0001) # Don't reduce or increase learning rate further. 
# The above two hyperparameters are good. 

huber_loss = keras.losses.Huber() 
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
eps = np.finfo(np.float32).eps.item()
#model = SAC("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10000, log_interval=1)
#model.save("sac_beer_game")

#del model # remove to demonstrate saving and loading

#model = SAC.load("sac_beer_game")
"""
_ = env.reset()
obs = env.observation_space.sample()
obs = np.squeeze(obs)
obs = tf.convert_to_tensor(obs)
obs = tf.expand_dims(obs, 0)
action_probs, critic_value = model(obs)
critic_value_history.append(critic_value[0, 0])

action = np.random.choice(num_actions, p=np.squeeze(action_probs))
action_probs_history.append(tf.math.log(action_probs[0, action]))
action = np.expand_dims(action, axis=0)

obs, reward, done, info = env.step(action)
obs = np.asarray(obs, dtype=np.float32)
print(obs, type(obs))
obs = tf.convert_to_tensor(obs)
print(obs, type(obs))
rewards_history.append(reward)
_ = env.reset()
"""
loss_track = []
reward_track = []
while True:
    _ = env.reset()
    obs = env.observation_space.sample()
    done = False
    episode_reward = 0
    
    obs = np.squeeze(obs)
    
    old_reward = 0
    with tf.GradientTape() as tape:
        while not done:    
            obs = tf.convert_to_tensor(obs)
            obs = tf.expand_dims(obs, 0)
            # This is a dummy normalization it works well. But actully you need to 
            # normalize the space where you finally return it from your code. 
            #obs = obs/1000
            action_probs, critic_value = model(obs)
            critic_value_history.append(critic_value[0, 0])
            
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            action = np.expand_dims(action, axis=0)
            print("Action_taken:", action)
            obs, curr_reward, done, info = env.step(action)
            reward = old_reward - curr_reward
            old_reward = curr_reward
            print("Reward...: ", reward)
            print(obs)
            obs = np.asarray(obs, dtype=np.float32)
            rewards_history.append(reward)
            episode_reward += reward        
            env.render()
        env.reset()
        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        # Running reward is tracked this gives the actual progress over time. 
        reward_track.append(running_reward)
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:            
            '''
            at this point, the critic estimates a total reward = 'value' in the future
            the agent takes an action with log probability = 'log_prob', and receives a reward = 'ret'
            the actor must then be updated to predict an action that leads t
            '''
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Loss is saved at this point
        loss_track.append(loss_value)
        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    # create a dataframe of each game
    df = env.df

    # Normal saving at every time step
    model.save_weights("Actor_Critic.h5")
    if running_reward > 5:  # Condition to consider the task solved
        # 
        print("Solved at episode {}!".format(episode_count))
        # Conditional saving, that means if a desired rewarad is obtained then 
        # save the model and it will be the best model that can be deployed and tested. 
        model.save_weights("Final_Weights_Actor_Critic.h5")
        break

# You can change the directory if you want, it will save in current program directory 
np.save("Reward_Plot_data.npy", reward_track, allow_pickle=True)
np.save("Loss_Plot_data.npy", loss_track, allow_pickle=True)

import matplotlib.pyplot as plt 
loss_p = np.load("Loss_Plot_data.npy", allow_pickle=True)
rew_p  = np.load("Reward_Plot_data.npy", allow_pickle=True)

plt.figure(figsize=((10,8)), dpi=100) # Change dpi to 300 for high resolution images. 

plt.plot(loss_p,"*-")
plt.xlabel("Episodes", fontsize=15)
plt.ylabel("Training Loss", fontsize=15)
plt.legend(["Actor Critic Loss"], loc="upper right", fontsize=15)
#plt.xlim(0,1400)  # x limit is between number of espisodes, if you run 100 episodes it will be (0,100)
#plt.xticks(np.arange(0,1401,200), fontsize=15) # for 100 episodes (np.arange(0,101,10), fontsize=15)
#plt.yticks(np.arange(0,700001,100000), fontsize=15) # It depends how maximum loss goes. So set the limit accordingly or don't add it 
#plt.ylim(0,700000) # Same as X limit but depends on maximum and minimum value of loss. 

# if you want to save the image as png transparent then uncomment the following line. 
# and comment the plt.show() otherwise you will not see any image. 
#plt.savefig("Loss_Plot.png", transparent=True)



plt.figure(figsize=((10,8)), dpi=100) # Change dpi to 300 for high resolution images. 

plt.plot(loss_p,"*-")
plt.xlabel("Episodes", fontsize=15)
plt.ylabel("Training Reward", fontsize=15)
plt.legend(["Actor Critic Reward"], loc="upper right", fontsize=15)
#plt.xlim(0,1400)  # x limit is between number of espisodes, if you run 100 episodes it will be (0,100)
#plt.xticks(np.arange(0,1401,200), fontsize=15) # for 100 episodes (np.arange(0,101,10), fontsize=15)
#plt.yticks(np.arange(0,700001,100000), fontsize=15) # It depends how maximum reward goes. So set the limit accordingly or don't add it 
#plt.ylim(0,700000) # Same as X limit but depends on maximum and minimum value of reward. 

# if you want to save the image as png transparent then uncomment the following line. 
# and comment the plt.show() otherwise you will not see any image. 
#plt.savefig("Reward_Plot.png", transparent=True)

# plot the last game
df.plot(y=['orders_from_manufacturer','orders_from_distributor','orders_from_wholesaler','orders_from_retailer','market_demand']) 
plt.xlabel('Round')
plt.ylabel('Orders')
df.plot(y=['manufacturer_backorder','distributor_backorder','wholesaler_backorder','retailer_backorder']) 
plt.xlabel('Round')
plt.ylabel('Backorder (BEER)')
df.plot(y=['manufacturer_inventory','distributor_inventory','wholesaler_inventory','retailer_inventory']) 
plt.xlabel('Round')
plt.ylabel('Inventory (BEER)')
df.plot(y=['manufacturer_balance','distributor_balance','wholesaler_balance','retailer_balance'])
plt.xlabel('Round')
plt.ylabel('ETH Balance')
df.plot(y=['manufacturer_expenses','distributor_expenses','wholesaler_expenses','retailer_expenses'])
plt.xlabel('Round')
plt.ylabel('Profit and Loss (ETH)')
plt.show()