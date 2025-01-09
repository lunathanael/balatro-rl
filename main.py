import gymnasium as gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, MultiHeadAttention, LayerNormalization, Add, Flatten, Conv2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time
import tensorflow as tf
import random

import balatro_gym as bg
from collections import deque

import gymnasium as gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, MultiHeadAttention, LayerNormalization, Add, Flatten, Conv2D
from keras.optimizers import Adam

class PGAgent:
    def __init__(self):
        self.hand_size = 8
        self.num_suits = 4
        self.num_ranks = 13
        self.gamma = 0.95
        self.learning_rate = 0.001

        self.num_epochs = 4
        self.batch_size = 16
        
        # Memory buffers with fixed queue size
        self.queue_size = 10_000
        self.buffer = deque(maxlen=self.queue_size)

        self.model = self._build_model()
        
        # Update TensorBoard setup
        self.log_dir = "logs/pg_agent_" + time.strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.tensorboard = TensorBoard(log_dir=self.log_dir)
        self.tensorboard.set_model(self.model)
        self.episode_count = 0
        self.steps_this_episode = 0
        self.total_steps = 0

    def _build_model(self):
        # Input layers - both hand and deck as 4x13 grids
        hand = Input(shape=(self.num_suits, self.num_ranks, 1))    # 4x13x1 grid for hand
        deck = Input(shape=(self.num_suits, self.num_ranks, 1))    # 4x13x1 grid for deck
        game_state = Input(shape=(2,))                             # [num_hands, num_discards]
        
        # Process hand with convolutions
        hand_features = Conv2D(64, (2, 2), activation='relu', padding='same')(hand)
        hand_features = Conv2D(64, (2, 2), activation='relu', padding='same')(hand_features)
        hand_features = Flatten()(hand_features)
        
        # Process deck with convolutions
        deck_features = Conv2D(64, (2, 2), activation='relu', padding='same')(deck)
        deck_features = Conv2D(64, (2, 2), activation='relu', padding='same')(deck_features)
        deck_features = Flatten()(deck_features)
        
        # Combine features
        combined = Add()([
            Dense(256)(hand_features),
            Dense(256)(deck_features),
            Dense(256)(game_state)
        ])
        combined = Dense(256, activation='relu')(combined)
        combined = LayerNormalization()(combined)
        
        # Two separate heads for play/discard actions
        play_features = Dense(128, activation='relu')(combined)
        play_logits = Dense(self.num_suits * self.num_ranks, activation='sigmoid', name='play_cards')(play_features)
        play_count = Dense(5, activation='softmax', name='play_count')(play_features)
        
        discard_features = Dense(128, activation='relu')(combined)
        discard_logits = Dense(self.num_suits * self.num_ranks, activation='sigmoid', name='discard_cards')(discard_features)
        discard_count = Dense(5, activation='softmax', name='discard_count')(discard_features)

        play_prob = Dense(1, activation='sigmoid', name='play_prob')(
            Add()([play_features, discard_features])
        )
        
        model = Model(
            inputs=[hand, deck, game_state],
            outputs=[play_logits, play_count, discard_logits, discard_count, play_prob]
        )
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'play_cards': 'binary_crossentropy',
                'play_count': 'categorical_crossentropy',
                'discard_cards': 'binary_crossentropy',
                'discard_count': 'categorical_crossentropy',
                'play_prob': 'binary_crossentropy',
            },
            loss_weights={
                'play_cards': 0.5,
                'play_count': 0.15,
                'discard_cards': 0.4,
                'discard_count': 0.1,
                'play_prob': 1.0,
            }
        )
        return model

    def preprocess_state(self, state):
        obs, info = state
        """Convert raw state into network inputs"""
        hand_grid = np.zeros((self.num_suits, self.num_ranks, 1))
        deck_grid = np.zeros((self.num_suits, self.num_ranks, 1))

        for card in info['state'].hand:
            hand_grid[int(card.suit), int(card.rank), 0] = 1
        for card in info['state'].deck:
            deck_grid[int(card.suit), int(card.rank), 0] = 1

            
        game_state = np.array([obs['hands'], obs['discards']])
        
        return [
            hand_grid[np.newaxis, ...],
            deck_grid[np.newaxis, ...],
            game_state[np.newaxis, ...]
        ]

    def act(self, state, mask=None, discards=None):
        """Select action based on state"""
        x = self.preprocess_state(state)
        play_cards, play_count, discard_cards, discard_count, play_prob = self.model.predict(x, verbose=0)
        
        if discards == 0:
            play_prob = 1

        # Choose action type based on highest probability
        if play_prob >= 0.5:
            num_cards = np.argmax(play_count[0]) + 1
            probs = play_cards[0]
            is_discard = False
        else:
            num_cards = np.argmax(discard_count[0]) + 1
            probs = discard_cards[0]
            is_discard = True

        if mask is not None:
            # Check for negative probabilities
            probs = np.maximum(probs, 0)
            probs[~mask] = -1
            

        selected_cards = np.argsort(probs)[-num_cards:]
        action_mask = np.zeros(self.num_suits * self.num_ranks, dtype=bool)
        action_mask[selected_cards] = True
        
        # Store action info for training
        return {
            'action_mask': action_mask,
            'is_discard': is_discard,
            'probs': probs,
            'num_cards': num_cards
        }

    def remember(self, trajectories):
        """Store and process trajectories (state-reward pairs) for training"""
        episode_states = []
        episode_rewards = []
        episode_actions = []
        
        # Collect episode data
        for state, action, reward in trajectories:
            episode_states.append(self.preprocess_state(state))
            episode_actions.append(action)
            episode_rewards.append(reward)
        
        # Calculate discounted rewards for this episode
        episode_rewards = np.array(episode_rewards)
        discounted = np.zeros_like(episode_rewards)
        running_add = 0
        
        for t in reversed(range(len(episode_rewards))):
            running_add = running_add * self.gamma + episode_rewards[t]
            discounted[t] = running_add
        
        # Normalize the rewards
        if len(discounted) > 1:  # Only normalize if episode has multiple steps
            discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)
        
        # Store processed trajectories
        for state, action, reward, discounted_reward in zip(episode_states, episode_actions, episode_rewards, discounted):
            self.buffer.append((state, action, reward, discounted_reward))

    def train(self):
        """Train the model on collected experience"""
        if len(self.buffer) < self.batch_size:
            return

        total_loss = 0
        for epoch in range(self.num_epochs):    
            indices = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)

            states = [self.buffer[i][0] for i in indices]
            actions = [self.buffer[i][1] for i in indices]
            rewards = [self.buffer[i][2] for i in indices]
            discounted_rewards = [self.buffer[i][3] for i in indices]

            states = [np.vstack([state[j] for state in states]) for j in range(3)]
            
            # Prepare targets
            play_cards = np.zeros((self.batch_size, 4*13))
            play_count = np.zeros((self.batch_size, 5))
            discard_cards = np.zeros((self.batch_size, 4*13))
            discard_count = np.zeros((self.batch_size, 5))
            play_prob = np.zeros((self.batch_size, 1))

            for i, (action, reward) in enumerate(zip(actions, discounted_rewards)):
                if action['is_discard']:
                    # Use original probabilities, scaled by reward
                    discard_cards[i][action['action_mask']] = action['probs'][action['action_mask']] * reward
                    discard_count[i][action['num_cards']-1] = 1  # One-hot encode the count
                    play_prob[i] = 0
                else:
                    # Use original probabilities, scaled by reward
                    play_cards[i][action['action_mask']] = action['probs'][action['action_mask']] * reward
                    play_count[i][action['num_cards']-1] = 1  # One-hot encode the count
                    play_prob[i] = 1

            # Train model and get loss values
            logs = self.model.train_on_batch(
                states,
                {
                    'play_cards': play_cards,
                    'play_count': play_count,
                    'discard_cards': discard_cards,
                    'discard_count': discard_count,
                    'play_prob': play_prob,
                }
            )
            
            # Track average loss per epoch
            total_loss += logs[0] if isinstance(logs, list) else logs
            
            # Log detailed training metrics
            with self.summary_writer.as_default():
                tf.summary.scalar('train/loss_per_epoch', logs[0] if isinstance(logs, list) else logs, step=self.episode_count)
                if isinstance(logs, list):
                    tf.summary.scalar('train/play_cards_loss', logs[1], step=self.episode_count)
                    tf.summary.scalar('train/play_count_loss', logs[2], step=self.episode_count)
                    tf.summary.scalar('train/discard_cards_loss', logs[3], step=self.episode_count)
                    tf.summary.scalar('train/discard_count_loss', logs[4], step=self.episode_count)
                    tf.summary.scalar('train/play_prob_loss', logs[5], step=self.episode_count)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make("Balatro-v0", render_mode="human")
    agent = PGAgent()
    
    # Add total actions counter
    total_actions = 0
    
    for episode in range(1_000_000):
        state = env.reset()
        episode_reward = 0
        agent.steps_this_episode = 0
        trajectories = []
        episode_actions = 0  # Track actions for this episode
        
        while True:
            valid_mask = state[0]['hand'].flatten()
            discards = state[0]['discards']
            action = agent.act(state, valid_mask, discards)

            action_mask = action['action_mask']
            is_discard = action['is_discard']
            obs, reward, done, truncated, info = env.step((action_mask, is_discard))
            
            trajectories.append((state, action, reward))
            agent.steps_this_episode += 1
            agent.total_steps += 1
            episode_actions += 1  # Increment action counter
            total_actions += 1    # Increment total actions
            episode_reward += reward
            state = (obs, info)
            
            if done:
                break
        
        agent.remember(trajectories)
        agent.train()
        agent.episode_count += 1
        
        # Calculate rewards per action
        reward_per_action = episode_reward / episode_actions if episode_actions > 0 else 0
        
        # Enhanced TensorBoard logging
        with agent.summary_writer.as_default():
            tf.summary.scalar('rewards/episode_total', episode_reward, step=episode)
            tf.summary.scalar('rewards/per_action', reward_per_action, step=episode)
            tf.summary.scalar('stats/training_steps', agent.total_steps, step=episode)
            tf.summary.scalar('stats/total_actions', total_actions, step=episode)
            tf.summary.scalar('stats/episode_actions', episode_actions, step=episode)
            tf.summary.scalar('stats/buffer_size', len(agent.buffer), step=episode)
            agent.summary_writer.flush()
        
        print(f"Episode {episode}: Reward = {episode_reward}, Actions = {episode_actions}, Reward/Action = {reward_per_action:.3f}")
        
        if episode % 10 == 0:
            agent.save('save_model/balatro_agent.weights.h5')