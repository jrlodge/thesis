from the_beer_game_environment import BeerGameEnv
from stable_baselines3 import SAC
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

num_inputs = 8
num_actions = 100
num_hidden = 64

    
inputs = layers.Input(shape=(num_inputs,))
dense = layers.Dense(64, activation="relu")(inputs)
dense = layers.Dense(64, activation="relu")(dense)
dense = layers.Dense(64, activation="relu")(dense)
dense = layers.Dense(64, activation="relu")(dense)
dense = layers.Dense(64, activation="relu")(dense)
action = layers.Dense(num_actions, activation="softmax")(dense)
critic = layers.Dense(1)(dense)

model = keras.Model(inputs=inputs, outputs=[action, critic])



env = BeerGameEnv()

gamma = 0.99
optimizer = keras.optimizers.Adam(learning_rate=0.01)
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

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 50000:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break