# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np
#
# class Critic(tf.keras.Model):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.l1 = layers.Dense(256, activation='relu')
#         self.l2 = layers.Dense(256, activation='relu')
#         self.q = layers.Dense(1)
#
#     def call(self, state_action):
#         x = self.l1(state_action)
#         x = self.l2(x)
#         q = self.q(x)
#         return q
#
# class Actor(tf.keras.Model):
#     def __init__(self, action_dim):
#         super(Actor, self).__init__()
#         self.l1 = layers.Dense(256, activation='relu')
#         self.l2 = layers.Dense(256, activation='relu')
#         self.mu = layers.Dense(action_dim)
#         self.log_sigma = layers.Dense(action_dim)
#
#     def call(self, state):
#         x = self.l1(state)
#         x = self.l2(x)
#         mu = self.mu(x)
#         log_sigma = self.log_sigma(x)
#         log_sigma = tf.clip_by_value(log_sigma, -20, 2)  # Log standard deviation
#         sigma = tf.exp(log_sigma)
#         return mu, sigma
#
# class SACAgent:
#     def __init__(self, state_dim, action_dim, max_action,
#                  lr=3e-4,
#                  discount=0.99, tau=0.005):
#         self.actor = Actor(action_dim)
#         self.critic1 = Critic()
#         self.critic2 = Critic()
#         self.target_critic1 = Critic()
#         self.target_critic2 = Critic()
#
#         # 强制初始化所有模型的权重
#         dummy_state = tf.convert_to_tensor(np.zeros((1, state_dim)), dtype=tf.float32)
#         dummy_action = tf.convert_to_tensor(np.zeros((1, action_dim)), dtype=tf.float32)
#         self.actor(dummy_state)
#         self.critic1(tf.concat([dummy_state, dummy_action], axis=1))
#         self.critic2(tf.concat([dummy_state, dummy_action], axis=1))
#         self.target_critic1(tf.concat([dummy_state, dummy_action], axis=1))
#         self.target_critic2(tf.concat([dummy_state, dummy_action], axis=1))
#
#         # Initialize optimizers
#         self.actor_optimizer = tf.keras.optimizers.Adam(lr)
#         self.critic_optimizer = tf.keras.optimizers.Adam(lr)
#         self.alpha_optimizer = tf.keras.optimizers.Adam(lr)
#
#         self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
#         self.alpha = tf.exp(self.log_alpha)
#         self.gamma = discount
#         self.tau = tau
#         self.action_dim = action_dim
#         self.max_action = max_action
#
#         # Copy weights from critic to target critic
#         self.target_critic1.set_weights(self.critic1.get_weights())
#         self.target_critic2.set_weights(self.critic2.get_weights())
#
#     def select_action(self, state):
#         state = tf.convert_to_tensor([state], dtype=tf.float32)
#         mu, sigma = self.actor(state)
#         dist = tf.random.normal(shape=mu.shape)
#         action = mu + sigma * dist
#         action = tf.tanh(action) * self.max_action
#         return action.numpy()[0]
#
#     def train(self, replay_buffer, batch_size=256):
#         # Sample a batch of transitions
#         state, action, next_state, reward, done = replay_buffer.sample(
#             batch_size)
#         state = tf.convert_to_tensor(state, dtype=tf.float32)
#         action = tf.convert_to_tensor(action, dtype=tf.float32)
#         next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
#         reward = tf.convert_to_tensor(reward, dtype=tf.float32)
#         done = tf.convert_to_tensor(done, dtype=tf.float32)
#
#         # Train the critic networks
#         with tf.GradientTape(persistent=True) as tape:
#             next_action = self.actor(next_state)
#             next_action = tf.random.normal(shape=next_action[0].shape) * \
#                           next_action[1] + next_action[0]
#             next_action = tf.tanh(next_action) * self.max_action
#
#             target_q1 = self.target_critic1(
#                 tf.concat([next_state, next_action], axis=1))
#             target_q2 = self.target_critic2(
#                 tf.concat([next_state, next_action], axis=1))
#             target_q = reward + (1.0 - done) * self.gamma * tf.minimum(
#                 target_q1, target_q2)
#
#             current_q1 = self.critic1(
#                 tf.concat([state, action], axis=1))
#             current_q2 = self.critic2(
#                 tf.concat([state, action], axis=1))
#
#             critic_loss1 = tf.keras.losses.MSE(target_q, current_q1)
#             critic_loss2 = tf.keras.losses.MSE(target_q, current_q2)
#
#         # Compute gradients for Critic
#         critic_grads1 = tape.gradient(critic_loss1,
#                                       self.critic1.trainable_variables)
#         critic_grads2 = tape.gradient(critic_loss2,
#                                       self.critic2.trainable_variables)
#
#         self.critic_optimizer.apply_gradients(
#             zip(critic_grads1, self.critic1.trainable_variables))
#         self.critic_optimizer.apply_gradients(
#             zip(critic_grads2, self.critic2.trainable_variables))
#
#         # Train the actor network
#         with tf.GradientTape() as tape:
#             mu, sigma = self.actor(state)
#             dist = tf.random.normal(shape=mu.shape)
#             action = mu + sigma * dist
#             action = tf.tanh(action) * self.max_action
#
#             log_prob = -0.5 * (dist ** 2 + 2 * tf.math.log(
#                 sigma) + tf.math.log(2 * np.pi))
#             log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
#             q1 = self.critic1(tf.concat([state, action], axis=1))
#             q2 = self.critic2(tf.concat([state, action], axis=1))
#             q = tf.minimum(q1, q2)
#             actor_loss = tf.reduce_mean(self.alpha * log_prob - q)
#
#         # Compute gradients for Actor
#         actor_grads = tape.gradient(actor_loss,
#                                     self.actor.trainable_variables)
#         self.actor_optimizer.apply_gradients(
#             zip(actor_grads, self.actor.trainable_variables))
#
#         # Update the temperature parameter
#         with tf.GradientTape() as tape:
#             log_prob = -0.5 * (dist ** 2 + 2 * tf.math.log(
#                 sigma) + tf.math.log(2 * np.pi))
#             log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
#             alpha_loss = -tf.reduce_mean(
#                 self.log_alpha * (log_prob + self.target_entropy))
#
#         # Compute gradients for Alpha
#         alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
#         self.alpha_optimizer.apply_gradients(
#             zip(alpha_grads, [self.log_alpha]))
#         self.alpha = tf.exp(self.log_alpha)
#
#         # Soft update of the target networks
#         for (target_param, param) in zip(
#                 self.target_critic1.trainable_variables,
#                 self.critic1.trainable_variables):
#             target_param.assign(
#                 self.tau * param + (1 - self.tau) * target_param)
#
#         for (target_param, param) in zip(
#                 self.target_critic2.trainable_variables,
#                 self.critic2.trainable_variables):
#             target_param.assign(
#                 self.tau * param + (1 - self.tau) * target_param)
#
# class ReplayBuffer:
#     def __init__(self, max_size=1000000):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         self.state = []
#         self.action = []
#         self.next_state = []
#         self.reward = []
#         self.done = []
#
#     def add(self, state, action, next_state, reward, done):
#         if self.size < self.max_size:
#             self.state.append(state)
#             self.action.append(action)
#             self.next_state.append(next_state)
#             self.reward.append(reward)
#             self.done.append(done)
#             self.size += 1
#         else:
#             self.state[self.ptr] = state
#             self.action[self.ptr] = action
#             self.next_state[self.ptr] = next_state
#             self.reward[self.ptr] = reward
#             self.done[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#         return (
#             np.array(self.state)[ind],
#             np.array(self.action)[ind],
#             np.array(self.next_state)[ind],
#             np.array(self.reward)[ind],
#             np.array(self.done)[ind]
#         )
#
#     def __len__(self):
#         return self.size
