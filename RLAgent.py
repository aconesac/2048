import tensorflow as tf
from keras.models import Sequential, load_model, clone_model   # type: ignore
from keras.layers import Dense, InputLayer, Dropout, BatchNormalization  # type: ignore
from keras.optimizers import Adam  # type: ignore
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, model = None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        if model:
            self.main_model = model
            print(type(self.main_model))
            self.target_model = clone_model(model)
            self.target_model.set_weights(self.main_model.get_weights())
            print(type(self.target_model))
        else:
            self.main_model = self._build_big_model()
            self.target_model = self._build_big_model()
            self.target_model.set_weights(self.main_model.get_weights())

    def _build_model(self) -> Sequential:
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(shape=(self.state_size,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        return model

    def _build_big_model(self):
        model = Sequential()
        model.add(InputLayer(shape=(self.state_size,)))
        model.add(BatchNormalization())
        model.add(Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
        model.add(Dropout(0.1))
        model.add(Dense(self.action_size, activation='linear'))

        return model
        
    def remember(self, state, action, reward, next_state, done):
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
        
    def preprocess_state(self, state):
        # Convert tile values to log2 representation (except 0)
        processed_state = np.zeros_like(state, dtype=np.float32)
        for i in range(state.shape[0]):
            if state[i] > 0:
                processed_state[i] = np.log2(state[i])

        # Normalize dynamically so values stay in [0, 1] even past 2048
        max_log = processed_state.max()
        if max_log > 0:
            processed_state = processed_state / max_log

        return processed_state

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = self.preprocess_state(state)
        act_values = self.main_model(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])  # returns action
    
    def sample_memory(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        for i in idx:
            elem = self.memory[i]
            state, action, reward, next_state, done = elem
            states.append(np.asarray(state))
            actions.append(np.asarray(action))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
            
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer."""
        # Calculate targets.
        next_qs = self.target_model(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        target = rewards + (1. - dones) * self.gamma * max_next_qs
        with tf.GradientTape() as tape:
            qs = self.main_model(states)
            action_masks = tf.one_hot(actions, self.action_size)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.mse(target, masked_qs)
        grads = tape.gradient(loss, self.main_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_variables))
        return loss
            
    def target_train(self):
        self.target_model.set_weights(self.main_model.get_weights())
        
    def load(self, name):
        self.main_model = load_model(name)
        self.target_model = load_model(name)

    def save(self, name):
        self.main_model.save(name)
