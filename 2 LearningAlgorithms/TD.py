import tensorflow as tf
import numpy as np
import random
from collections import deque
from tqdm import tqdm

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = tf.keras.layers.Dense(400, activation='relu')
        self.layer2 = tf.keras.layers.Dense(300, activation='relu')
        self.layer3 = tf.keras.layers.Dense(300, activation='relu')
        self.layer4 = tf.keras.layers.Dense(action_dim, activation='tanh')
        self.max_action = max_action

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x * self.max_action

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = tf.keras.layers.Dense(400, activation='relu')
        self.layer2 = tf.keras.layers.Dense(300, activation='relu')
        self.layer3 = tf.keras.layers.Dense(300, activation='relu')
        self.layer4 = tf.keras.layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class TD3:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 actor_lr=0.0001,
                 critic_lr=0.0001,
                 replay_buffer=2000,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 batch_size=32):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.set_weights(self.actor.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_target_1 = Critic(state_dim, action_dim)
        self.critic_target_2 = Critic(state_dim, action_dim)
        self.critic_target_1.set_weights(self.critic_1.get_weights())
        self.critic_target_2.set_weights(self.critic_2.get_weights())

        self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.replay_buffer = deque(maxlen=replay_buffer)
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

    def select_action(self, state):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.actor(state).numpy().flatten()

    # 在train方法中调试
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, next_state, reward, done = map(np.stack,
                                                      zip(*batch))

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.reshape(
            tf.convert_to_tensor(reward, dtype=tf.float32), (-1, 1))
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        done = tf.reshape(done, (-1, 1))

        # noise = tf.clip_by_value(tf.random.normal(shape=action.shape,
        #                                           stddev=self.policy_noise),
        #                          -self.noise_clip, self.noise_clip)

        # print("next_state shape:", next_state.shape)
        next_action = self.actor_target(next_state)
        # print("self.actor_target(next_state) shape:", next_action.shape)
        # print("noise shape:", noise.shape)

        next_action = tf.clip_by_value(next_action
                                       -self.actor.max_action,
                                       self.actor.max_action)

        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = tf.minimum(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.discount * target_Q

        with tf.GradientTape() as tape:
            current_Q1 = self.critic_1(state, action)
            critic_loss_1 = tf.keras.losses.MeanSquaredError()(target_Q,
                                                               current_Q1)
        critic_gradients_1 = tape.gradient(critic_loss_1,
                                           self.critic_1.trainable_variables)
        self.critic_optimizer_1.apply_gradients(
            zip(critic_gradients_1, self.critic_1.trainable_variables))

        with tf.GradientTape() as tape:
            current_Q2 = self.critic_2(state, action)
            critic_loss_2 = tf.keras.losses.MeanSquaredError()(target_Q,
                                                               current_Q2)
        critic_gradients_2 = tape.gradient(critic_loss_2,
                                           self.critic_2.trainable_variables)
        self.critic_optimizer_2.apply_gradients(
            zip(critic_gradients_2, self.critic_2.trainable_variables))

        if self.total_it % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                actor_loss = -tf.reduce_mean(
                    self.critic_1(state, self.actor(state)))
            actor_gradients = tape.gradient(actor_loss,
                                            self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_gradients, self.actor.trainable_variables))

            # 软更新目标网络
            new_weights_1 = []
            for param, target_param in zip(
                    self.critic_1.trainable_variables,
                    self.critic_target_1.trainable_variables):
                new_weights_1.append(
                    self.tau * param + (1 - self.tau) * target_param)
            self.critic_target_1.set_weights(new_weights_1)

            new_weights_2 = []
            for param, target_param in zip(
                    self.critic_2.trainable_variables,
                    self.critic_target_2.trainable_variables):
                new_weights_2.append(
                    self.tau * param + (1 - self.tau) * target_param)
            self.critic_target_2.set_weights(new_weights_2)

            new_weights_actor = []
            for param, target_param in zip(
                    self.actor.trainable_variables,
                    self.actor_target.trainable_variables):
                new_weights_actor.append(
                    self.tau * param + (1 - self.tau) * target_param)
            self.actor_target.set_weights(new_weights_actor)

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

def train_td3(env, agent, num_episodes):
    episode_rewards = []
    episode_surveillance_rate = []
    episode_steps = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        sur_count = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()
            state = next_state
            episode_reward += reward

            if done:
                step_count = info["count"]
                sur_count = info["sur_count"]
                max_continuous_hit = info["max_continuous_hit"]

        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        episode_surveillance_rate.append(sur_count / step_count)
        print(f"{episode = }, {episode_reward = }, surveillance rate = {sur_count}/{step_count}, {max_continuous_hit = }")

    return episode_rewards, episode_surveillance_rate
