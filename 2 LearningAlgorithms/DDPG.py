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

class DDPG:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 actor_lr=0.0001,
                 critic_lr=0.0001,
                 replay_buffer=2000,
                 discount=0.99,
                 tau=0.005,
                 batch_size=32,
                 actor_optimizer='adam',
                 critic_optimizer='adam'):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.set_weights(self.actor.get_weights())

        if actor_optimizer == 'adam':
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        elif actor_optimizer == 'adagrad':
            self.actor_optimizer = tf.keras.optimizers.Adagrad(learning_rate=actor_lr)
        elif actor_optimizer == 'sgd':
            self.actor_optimizer = tf.keras.optimizers.SGD(learning_rate=actor_lr, momentum=0.9)
        elif actor_optimizer == 'rmsprop':
            self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=actor_lr)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.set_weights(self.critic.get_weights())

        if critic_optimizer == 'adam':
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        elif critic_optimizer == 'adagrad':
            self.critic_optimizer = tf.keras.optimizers.Adagrad(learning_rate=critic_lr)
        elif critic_optimizer == 'sgd':
            self.critic_optimizer = tf.keras.optimizers.SGD(learning_rate=critic_lr, momentum=0.9)
        elif critic_optimizer == 'rmsprop':
            self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=critic_lr)

        self.replay_buffer = deque(maxlen=replay_buffer)
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.actor(state).numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.reshape(tf.convert_to_tensor(reward, dtype=tf.float32), (-1, 1))
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        done = tf.reshape(done, (-1, 1))

        with tf.GradientTape() as tape:
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (1 - done) * self.discount * target_Q
            current_Q = self.critic(state, action)
            critic_loss = tf.keras.losses.MeanSquaredError()(target_Q, current_Q)

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(self.critic(state, self.actor(state)))

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # 软更新目标网络
        new_weights = []
        for param, target_param in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
            new_weights.append(self.tau * param + (1 - self.tau) * target_param)
        self.critic_target.set_weights(new_weights)

        new_weights = []
        for param, target_param in zip(self.actor.trainable_variables, self.actor_target.trainable_variables):
            new_weights.append(self.tau * param + (1 - self.tau) * target_param)
        self.actor_target.set_weights(new_weights)

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))


def train_ddpg(env, agent, num_episodes):
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
        episode_surveillance_rate.append(sur_count/step_count)
        print(f"{episode = }, {episode_reward = }, surveillance rate = {sur_count}/{step_count}, {max_continuous_hit = }")

    return episode_rewards, episode_surveillance_rate


# 注意：训练完之后，可以将模型转换为TFLite格式
def convert_to_tflite(model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(model_name, 'wb') as f:
        f.write(tflite_model)

# 例子：转换actor模型为TFLite格式
# convert_to_tflite(agent.actor, 'actor_model.tflite')
