import gymnasium as gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, MultiHeadAttention, LayerNormalization, Add, Flatten, Conv2D
from keras.optimizers import Adam

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
        self.gamma = 0.99
        self.learning_rate = 0.001
        
        # Memory buffers with fixed queue size
        self.queue_size = 1000
        self.states = deque(maxlen=self.queue_size)
        self.actions = deque(maxlen=self.queue_size)
        self.rewards = deque(maxlen=self.queue_size)
        
        self.model = self._build_model()
        
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
                'play_cards': 1.0,
                'play_count': 1.0,
                'discard_cards': 1.0,
                'discard_count': 1.0,
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

        # Convert hand to 4x13 grid
        # hand_grid = np.zeros((self.num_suits, self.num_ranks, 1))
        # hand_cards = np.where(state['hand'] == 1)[0]
        # for card in hand_cards:
        #     rank = card % 13
        #     suit = card // 13
        #     hand_grid[suit, rank, 0] = 1
            
        # # Convert deck to 4x13 grid
        # deck_grid = np.zeros((self.num_suits, self.num_ranks, 1))
        # deck_cards = np.where(state['deck'] == 1)[0]
        # for card in deck_cards:
        #     rank = card % 13
        #     suit = card // 13
        #     deck_grid[suit, rank, 0] = 1
            
        game_state = np.array([obs['hands'], obs['discards']])
        
        return [
            hand_grid[np.newaxis, ...],
            deck_grid[np.newaxis, ...],
            game_state[np.newaxis, ...]
        ]

    def act(self, state, mask=None):
        """Select action based on state"""
        x = self.preprocess_state(state)
        play_cards, play_count, discard_cards, discard_count, play_prob = self.model.predict(x, verbose=0)
        
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
            probs[~mask] = 0
            
        # Select cards based on probabilities
        selected_cards = np.argsort(probs)[-num_cards:]
        action_mask = np.zeros(self.num_suits * self.num_ranks, dtype=bool)
        action_mask[selected_cards] = True
        
        # Store action info for training
        self.actions.append({
            'mask': action_mask,
            'is_discard': is_discard,
            'probs': probs,
            'num_cards': num_cards
        })
        
        return action_mask, is_discard

    def remember(self, state, reward):
        """Store state and reward for training"""
        self.states.append(self.preprocess_state(state))
        self.rewards.append(reward)

    def discount_rewards(self):
        """Calculate discounted rewards"""
        rewards = np.array(self.rewards)
        discounted = np.zeros_like(rewards)
        running_add = 0
        
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
            
        # Normalize rewards
        discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)
        return discounted

    def train(self):
        """Train the model on collected experience"""
        if len(self.states) == 0:
            return
            
        # Calculate discounted rewards
        discounted_rewards = self.discount_rewards()
        
        # Prepare training data
        states = [np.vstack([s[i] for s in self.states]) for i in range(3)]
        
        # Prepare targets
        play_cards = np.zeros((len(self.actions), 4*13))
        play_count = np.zeros((len(self.actions), 5))
        discard_cards = np.zeros((len(self.actions), 4*13))
        discard_count = np.zeros((len(self.actions), 5))
        play_prob = np.zeros((len(self.actions), 1))

        for i, (action, reward) in enumerate(zip(self.actions, discounted_rewards)):
            if action['is_discard']:
                # Use original probabilities, scaled by reward
                discard_cards[i][action['mask']] = action['probs'][action['mask']] * reward
                discard_count[i][action['num_cards']-1] = 1  # One-hot encode the count
                play_prob[i] = 0
            else:
                # Use original probabilities, scaled by reward
                play_cards[i][action['mask']] = action['probs'][action['mask']] * reward
                play_count[i][action['num_cards']-1] = 1  # One-hot encode the count
                play_prob[i] = 1
        # Train model
        self.model.train_on_batch(
            states,
            {
                'play_cards': play_cards,
                'play_count': play_count,
                'discard_cards': discard_cards,
                'discard_count': discard_count,
                'play_prob': play_prob,
            }
        )
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make("Balatro-v0", render_mode="human")
    agent = PGAgent()
    
    for episode in range(100_000):
        state = env.reset()
        episode_reward = 0
        
        while True:
            valid_mask = state[0]['hand'].flatten()
            action_mask, is_discard = agent.act(state, valid_mask)
            obs, reward, done, truncated, info = env.step((action_mask, is_discard))
            
            agent.remember(state, reward)
            episode_reward += reward
            state = (obs, info)
            
            if done:
                break
        
        agent.train()
        print(f"Episode {episode}: Reward = {episode_reward}")
        
        if episode % 10 == 0:
            agent.save('save_model/balatro_agent.weights.h5')