import balatro_gym as bg

game = bg.core.GameState(42069)

def dfs(state, depth=0, max_depth=3):
    if depth >= max_depth or state.is_terminal():
        return state.score, []
    
    best_score = -float('inf')
    best_moves = []
    
    # Try all possible combinations of cards (up to 5 cards)
    hand = state.hand.cards
    n = len(hand)
    
    for num_cards in range(1, min(6, n+1)):
        # Generate all combinations of num_cards from hand
        for i in range(1 << n):
            if bin(i).count('1') != num_cards:
                continue
                
            # Create action integer based on selected cards
            action = 0
            selected_cards = []
            for j in range(n):
                if i & (1 << j):
                    action |= (1 << j)
                    selected_cards.append(hand[j])
            
            # Create new state and apply move
            new_state = bg.core.GameState()
            new_state.hand = bg.core.Hand(list(state.hand.cards))
            new_state.deck = list(state.deck)
            new_state.score = state.score
            new_state.hands = state.hands
            new_state.discards = state.discards
            reward = new_state.play(action)
            
            # Recursively search
            score, moves = dfs(new_state, depth + 1, max_depth)
            score += reward
            
            if score > best_score:
                best_score = score
                best_moves = [(selected_cards, reward)] + moves
                if depth == 0:
                    print(f"Best score found: {best_score}")
                    print("\nBest move sequence:")
                    for cards, reward in best_moves:
                        print(f"Play {cards} for {reward} points")
    
    return best_score, best_moves

# Find best sequence of moves
final_score, move_sequence = dfs(game)
print(f"Best score found: {final_score}")
print("\nBest move sequence:")
for cards, reward in move_sequence:
    print(f"Play {cards} for {reward} points")
