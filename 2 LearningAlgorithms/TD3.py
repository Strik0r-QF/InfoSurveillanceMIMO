# import tensorflow as tf
# import random
# from tensorflow.keras import layers
# import numpy as np
#
#
# class Actor(tf.keras.Model):
#     def __init__(self, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.l1 = layers.Dense(400, activation='relu')
#         self.l2 = layers.Dense(300, activation='relu')
#         self.l3 = layers.Dense(action_dim, activation='tanh')
#         self.max_action = max_action
#
#     def call(self, state):
#         x = self.l1(state)
#         x = self.l2(x)
#         return self.max_action * self.l3(x)
#
#
# class Critic(tf.keras.Model):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.l1 = layers.Dense(400, activation='relu')
#         self.l2 = layers.Dense(300, activation='relu')
#         self.q = layers.Dense(1)
#
#     def call(self, state_action):
#         x = self.l1(state_action)
#         x = self.l2(x)
#         return self.q(x)
#
#
# class TD3Agent:
#     def __init__(self, state_dim, action_dim, action_bound,
#                  gamma=0.99, tau=0.005, lr=0.001, batch_size=32,
#                  replay_buffer=2000):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.action_bound = action_bound
#         self.gamma = gamma
#         self.tau = tau
#
#         self.actor_model = self.create_actor_model()
#         self.critic_model_1 = self.create_critic_model()
#         self.critic_model_2 = self.create_critic_model()
#
#         self.target_actor_model = self.create_actor_model()
#         self.target_critic_model_1 = self.create_critic_model()
#         self.target_critic_model_2 = self.create_critic_model()
#
#         self.target_actor_model.set_weights(self.actor_model.get_weights())
#         self.target_critic_model_1.set_weights(self.critic_model_1.get_weights())
#         self.target_critic_model_2.set_weights(self.critic_model_2.get_weights())
#
#         self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#         self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr)
#         self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr)
#
#         self.replay_buffer = []
#         self.buffer_capacity = replay_buffer
#         self.batch_size = batch_size
#
#     def create_actor_model(self):
#         model = tf.keras.Sequential()
#         model.add(layers.Input(shape=(self.state_dim,)))
#         model.add(layers.Dense(256, activation='relu'))
#         model.add(layers.Dense(256, activation='relu'))
#         model.add(layers.Dense(self.action_dim, activation='tanh'))
#         model.add(layers.Lambda(lambda x: x * self.action_bound))
#         return model
#
#     def create_critic_model(self):
#         state_input = layers.Input(shape=(self.state_dim,))
#         action_input = layers.Input(shape=(self.action_dim,))
#         concat = layers.Concatenate()([state_input, action_input])
#
#         out = layers.Dense(256, activation='relu')(concat)
#         out = layers.Dense(256, activation='relu')(out)
#         out = layers.Dense(1)(out)
#
#         model = tf.keras.Model(inputs=[state_input, action_input], outputs=out)
#         return model
#
#     def update_target(self, target_weights, weights, tau):
#         for (a, b) in zip(target_weights, weights):
#             a.assign(b * tau + a * (1 - tau))
#
#     def policy(self, state):
#         sampled_actions = tf.squeeze(self.actor_model(state))
#         sampled_actions = sampled_actions.numpy()
#         return np.clip(sampled_actions, -self.action_bound, self.action_bound)
#
#     def learn(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return
#
#         minibatch = random.sample(self.replay_buffer, self.batch_size)
#         state_batch = np.array([experience[0] for experience in minibatch])
#         action_batch = np.array([experience[1] for experience in minibatch])
#         reward_batch = np.array([experience[2] for experience in minibatch])
#         next_state_batch = np.array([experience[3] for experience in minibatch])
#         done_batch = np.array([experience[4] for experience in minibatch])
#
#         # 将 NumPy 数组转换为 TensorFlow 张量
#         state_batch = tf.convert_to_tensor(state_batch,
#                                            dtype=tf.float32)
#         action_batch = tf.convert_to_tensor(action_batch,
#                                             dtype=tf.float32)
#         reward_batch = tf.convert_to_tensor(reward_batch,
#                                             dtype=tf.float32)
#         next_state_batch = tf.convert_to_tensor(next_state_batch,
#                                                 dtype=tf.float32)
#         done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)
#
#         with tf.GradientTape(persistent=True) as tape:
#             target_actions = self.target_actor_model(next_state_batch)
#             target_noise = tf.random.normal(shape=target_actions.shape, stddev=0.2)
#             target_actions = tf.clip_by_value(target_actions + target_noise, -self.action_bound, self.action_bound)
#
#             target_q1 = self.target_critic_model_1([next_state_batch, target_actions])
#             target_q2 = self.target_critic_model_2([next_state_batch, target_actions])
#             target_q = reward_batch + self.gamma * tf.minimum(target_q1, target_q2) * (1 - done_batch)
#
#             current_q1 = self.critic_model_1([state_batch, action_batch])
#             current_q2 = self.critic_model_2([state_batch, action_batch])
#             critic_loss_1 = tf.reduce_mean(tf.square(current_q1 - target_q))
#             critic_loss_2 = tf.reduce_mean(tf.square(current_q2 - target_q))
#
#         critic_grad_1 = tape.gradient(critic_loss_1, self.critic_model_1.trainable_variables)
#         critic_grad_2 = tape.gradient(critic_loss_2, self.critic_model_2.trainable_variables)
#         self.critic_optimizer_1.apply_gradients(zip(critic_grad_1, self.critic_model_1.trainable_variables))
#         self.critic_optimizer_2.apply_gradients(zip(critic_grad_2, self.critic_model_2.trainable_variables))
#
#         if np.random.random() < 0.5:
#             with tf.GradientTape() as tape:
#                 actions = self.actor_model(state_batch)
#                 critic_value = self.critic_model_1([state_batch, actions])
#                 actor_loss = -tf.reduce_mean(critic_value)
#
#             actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
#             self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
#
#             self.update_target(self.target_actor_model.variables, self.actor_model.variables, self.tau)
#             self.update_target(self.target_critic_model_1.variables, self.critic_model_1.variables, self.tau)
#             self.update_target(self.target_critic_model_2.variables, self.critic_model_2.variables, self.tau)
#
#     def store_transition(self, state, action, reward, next_state, done):
#         self.replay_buffer.append((state, action, reward, next_state, done))
#         if len(self.replay_buffer) > self.buffer_capacity:
#             self.replay_buffer.pop(0)
#
# class ReplayBuffer:
#     def __init__(self, state_dim, action_dim, max_size=int(1e6)):
#         self.state = np.zeros((max_size, state_dim))
#         self.action = np.zeros((max_size, action_dim))
#         self.next_state = np.zeros((max_size, state_dim))
#         self.reward = np.zeros((max_size, 1))
#         self.not_done = np.zeros((max_size, 1))
#         self.ptr, self.size, self.max_size = 0, 0, max_size
#
#     def add(self, state, action, next_state, reward, done):
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1. - done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#         return (
#             tf.convert_to_tensor(self.state[ind], dtype=tf.float32),
#             tf.convert_to_tensor(self.action[ind], dtype=tf.float32),
#             tf.convert_to_tensor(self.next_state[ind], dtype=tf.float32),
#             tf.convert_to_tensor(self.reward[ind], dtype=tf.float32),
#             tf.convert_to_tensor(self.not_done[ind], dtype=tf.float32)
#         )