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
        self.gamma = 0.97
        self.learning_rate = 0.001

        self.num_epochs = 4
        self.batch_size = 32
        
        # Memory buffers with fixed queue size
        self.queue_size = 5_000
        self.buffer = deque(maxlen=self.queue_size)

        self.model = self._build_model()
        
        self.log_dir = "logs/pg_agent_97g_mse_weighted_" + time.strftime("%Y%m%d-%H%M%S")
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
        
        # Process hand with convolutions - using both 2x2 and 1x3 kernels for different patterns
        hand_features_2x2 = Conv2D(32, (2, 2), activation='relu', padding='same')(hand)
        hand_features_2x2 = Conv2D(32, (2, 2), activation='relu', padding='same')(hand_features_2x2)
        
        # 1x3 kernel for detecting straights (horizontal patterns)
        hand_features_1x3 = Conv2D(32, (1, 3), activation='relu', padding='same')(hand)
        hand_features_1x3 = Conv2D(32, (1, 3), activation='relu', padding='same')(hand_features_1x3)
        
        # 4x1 kernel for detecting flushes (vertical patterns)
        hand_features_4x1 = Conv2D(32, (4, 1), activation='relu', padding='same')(hand)
        hand_features_4x1 = Conv2D(32, (4, 1), activation='relu', padding='same')(hand_features_4x1)
        
        # Combine all features
        hand_features = Add()([hand_features_2x2, hand_features_1x3, hand_features_4x1])
        hand_features = Flatten()(hand_features)
        
        # Similar processing for deck
        deck_features_2x2 = Conv2D(32, (2, 2), activation='relu', padding='same')(deck)
        deck_features_2x2 = Conv2D(32, (2, 2), activation='relu', padding='same')(deck_features_2x2)
        
        deck_features_1x3 = Conv2D(32, (1, 3), activation='relu', padding='same')(deck)
        deck_features_1x3 = Conv2D(32, (1, 3), activation='relu', padding='same')(deck_features_1x3)
        
        deck_features_4x1 = Conv2D(32, (4, 1), activation='relu', padding='same')(deck)
        deck_features_4x1 = Conv2D(32, (4, 1), activation='relu', padding='same')(deck_features_4x1)
        
        deck_features = Add()([deck_features_2x2, deck_features_1x3, deck_features_4x1])
        deck_features = Flatten()(deck_features)
        
        card_features = Add()([
            Dense(256)(hand_features),
            Dense(256)(deck_features)
        ])
        card_features = Dense(256, activation='relu')(card_features)
        card_features = LayerNormalization()(card_features)
        
        combined = Add()([
            card_features,
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
                'play_prob': 'mse',
            },
            loss_weights={
                'play_cards': 1.0,
                'play_count': 0.05,
                'discard_cards': 1.0,
                'discard_count': 0.05,
                'play_prob': 500.0,
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
        
        if mask is None:
            mask = np.ones(52, dtype=np.bool)

        original_play_cards = play_cards.copy()
        original_play_count = play_count.copy()
        original_discard_cards = discard_cards.copy()
        original_discard_count = discard_count.copy()
        original_play_prob = play_prob.copy()

        play_prob = np.squeeze(play_prob)
        play_prob = np.clip(play_prob, 0, 1)

        if discards == 0:
            play_prob = 1
        
        action_type_probs = np.array([play_prob, 1-play_prob])
        action_type_probs = action_type_probs / np.sum(action_type_probs)
        action_type = np.random.choice([0, 1], p=action_type_probs)

        play_num_cards_probs = play_count[0]
        discard_num_cards_probs = discard_count[0]

        play_num_cards_probs = play_num_cards_probs / np.sum(play_num_cards_probs)
        discard_num_cards_probs = discard_num_cards_probs / np.sum(discard_num_cards_probs)

        play_num_cards = np.random.choice(len(play_count[0]), p=play_num_cards_probs) + 1
        discard_num_cards = np.random.choice(len(discard_count[0]), p=discard_num_cards_probs) + 1
        
        play_probs = play_cards[0]
        discard_probs = discard_cards[0]

        # Check for negative probabilities
        play_probs = np.maximum(play_probs, 0)
        discard_probs = np.maximum(discard_probs, 0)
        play_probs[~mask] = 0
        discard_probs[~mask] = 0

        if np.count_nonzero(play_probs) < play_num_cards:
            play_num_cards = np.count_nonzero(play_probs)
        if play_num_cards == 0:
            valid_indices = np.where(mask)[0]
            max_card = max(5, np.count_nonzero(mask))
            play_num_cards = np.random.randint(1, max_card)
            play_selected_cards = np.random.choice(valid_indices, size=play_num_cards, replace=False)
        else:
            play_probs = play_probs / np.sum(play_probs)
            play_selected_cards = np.random.choice(
                len(play_probs), 
                size=play_num_cards, 
                replace=False, 
                p=play_probs
            )

        if np.count_nonzero(discard_probs) < discard_num_cards:
            discard_num_cards = np.count_nonzero(discard_probs)
        if discard_num_cards == 0:
            valid_indices = np.where(mask)[0]
            max_card = max(5, np.count_nonzero(mask))
            discard_num_cards = np.random.randint(1, max_card)
            discard_selected_cards = np.random.choice(valid_indices, size=discard_num_cards, replace=False)
        else:
            discard_probs = discard_probs / np.sum(discard_probs)
            discard_selected_cards = np.random.choice(
                len(discard_probs), 
                size=discard_num_cards, 
                replace=False, 
                p=discard_probs
            )

        if action_type == 0:
            selected_cards = play_selected_cards
            is_discard = False
        else:
            selected_cards = discard_selected_cards
            is_discard = True
            
        action_mask = np.zeros(self.num_suits * self.num_ranks, dtype=bool)
        action_mask[selected_cards] = True
        
        return {
            'action_mask': action_mask,
            'play_prob': original_play_prob,
            'is_discard': is_discard,
            'play_probs': original_play_cards,
            'discard_probs': original_discard_cards,
            'play_num_cards': play_num_cards,
            'discard_num_cards': discard_num_cards,
            'play_num_cards_probs': original_play_count,
            'discard_num_cards_probs': original_discard_count
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
            discounted_rewards = discounted_rewards / np.std(discounted_rewards - np.mean(discounted_rewards))

            states = [np.vstack([state[j] for state in states]) for j in range(3)]
            
            # Prepare targets
            play_cards = np.zeros((self.batch_size, 4*13), dtype=np.float32)
            play_count = np.zeros((self.batch_size, 5), dtype=np.float32)
            discard_cards = np.zeros((self.batch_size, 4*13), dtype=np.float32)
            discard_count = np.zeros((self.batch_size, 5), dtype=np.float32)
            play_prob = np.zeros((self.batch_size, 1), dtype=np.float32)

            for i, (action, reward) in enumerate(zip(actions, discounted_rewards)):
                if action['is_discard']:
                    discard_cards[i][action['action_mask']] = 1
                    discard_count[i][action['discard_num_cards']-1] = 1  # One-hot encode the count
                    play_cards[i] = action['play_probs']
                    play_count[i] = action['play_num_cards_probs']
                    play_prob[i] = 0
                else:
                    play_cards[i][action['action_mask']] = 1
                    play_count[i][action['play_num_cards']-1] = 1  # One-hot encode the count
                    discard_cards[i] = action['discard_probs']
                    discard_count[i] = action['discard_num_cards_probs']
                    play_prob[i] = 1

                play_cards[i] = (np.array(play_cards[i]).astype('float32') - action['play_probs']) * reward
                discard_cards[i] = (np.array(discard_cards[i]).astype('float32') - action['discard_probs']) * reward
                play_count[i] = (np.array(play_count[i]).astype('float32') - action['play_num_cards_probs']) * reward
                discard_count[i] = (np.array(discard_count[i]).astype('float32') - action['discard_num_cards_probs']) * reward
                play_prob[i] = (np.array(play_prob[i]).astype('float32') - action['play_prob']) * reward
                
                play_cards[i] = action['play_probs'] + self.learning_rate * play_cards[i]
                discard_cards[i] = action['discard_probs'] + self.learning_rate * discard_cards[i]
                play_count[i] = action['play_num_cards_probs'] + self.learning_rate * play_count[i]
                discard_count[i] = action['discard_num_cards_probs'] + self.learning_rate * discard_count[i]
                play_prob[i] = action['play_prob'] + self.learning_rate * play_prob[i]


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


def generate_random_games(num_games):
    game_trajectories = []
    env = gym.make("Balatro-v0", render_mode="human")
    for _ in range(num_games):
        state = env.reset()
        episode_reward = 0
        trajectories = []
        while True:
            actual_hand = list(state[1]["state"].hand.cards)
            actual_hand = [card for card in actual_hand if card != bg.core.Cards.No_Card]

            mask = np.zeros(52, dtype=np.bool)
            for card in actual_hand:
                mask[bg.utils.card_to_index(card)] = 1
            card_probs = np.ones(52, dtype=np.float32) / len(actual_hand)
            card_probs[~mask] = 0

            num_cards_in_hand = len(actual_hand)
            num_cards_probs = np.ones(5, dtype=np.float32) / min(5, num_cards_in_hand)
            num_cards_probs[min(5, num_cards_in_hand):] = 0

            play_num_cards = np.random.choice(range(1,6), p=num_cards_probs)
            discard_num_cards = np.random.choice(range(1,6), p=num_cards_probs)

            is_discard = random.choice([True, False])
            play_prob = 0.5
            if state[0]['discards'] == 0:
                is_discard = False
                play_prob = 1

            discard_selected_cards = random.sample(actual_hand, discard_num_cards)
            play_selected_cards = random.sample(actual_hand, play_num_cards)
    
            play_playing_hand = np.zeros(52, dtype=np.bool)
            for card in play_selected_cards:
                play_playing_hand[bg.utils.card_to_index(card)] = 1
            discard_playing_hand = np.zeros(52, dtype=np.bool)
            for card in discard_selected_cards:
                discard_playing_hand[bg.utils.card_to_index(card)] = 1

            if is_discard:
                playing_hand = discard_playing_hand
            else:
                playing_hand = play_playing_hand

            action = (playing_hand, is_discard)

            obs, reward, done, truncated, info = env.step(action)

            probs = np.ones(52, dtype=np.float32) / np.sum(playing_hand)
            probs[~playing_hand] = 0

            action = {
                'action_mask': playing_hand,
                'is_discard': is_discard,
                'play_prob': play_prob,
                'play_probs': card_probs,
                'discard_probs': card_probs,
                'play_num_cards': play_num_cards,
                'discard_num_cards': discard_num_cards,
                'play_num_cards_probs': num_cards_probs,
                'discard_num_cards_probs': num_cards_probs
            }

            trajectories.append((state, action, reward))
            state = (obs, info)
            episode_reward += reward
            if done:
                break
        game_trajectories.append(trajectories)
    return game_trajectories

if __name__ == "__main__":
    env = gym.make("Balatro-v0", render_mode="human")
    agent = PGAgent()

    game_trajectories = generate_random_games(500)
    for trajectory in game_trajectories:
        agent.remember(trajectory)
    agent.train()
    
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
        
        if episode % 1000 == 0:
            agent.save('save_model/balatro_agent_97g_mse_weighted.weights.h5')
