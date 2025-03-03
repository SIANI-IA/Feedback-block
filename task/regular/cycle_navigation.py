import numpy as np

class CycleNavigation:
    """
    Cycle Navigation (R): Given a sequence of movements on a cycle of length 5,
    compute the end position. The movements are:
        - 0: STAY (no movement)
        - 1: INCREASE (move right)
        - 2: DECREASE (move left)
    
    The agent always starts at position 0. The final position is computed using
    modular arithmetic.
    
    By default, the cycle length is 5.
    """
    
    def __init__(self, cycle_length=5):
        self.cycle_length = cycle_length

    def sample_batch(self, amount: int, length: int) -> list:
        """
        Generates a batch of action sequences and their corresponding final positions.
        
        Args:
            amount (int): Number of sequences in the batch.
            length (int): Length of each action sequence.
        
        Returns:
            list: Each element is a string of format "<actions>=<final_position>"
        """
        rng = np.random.default_rng()
        actions = rng.choice([0, 1, 2], size=(amount, length))
        
        # Compute final positions (treating 2 as -1 for left movement)
        final_states = np.sum(np.where(actions == 2, -1, actions), axis=1) % self.cycle_length
        
        # Convert to formatted strings
        formatted_strings = [
            f"{''.join(map(str, actions[i]))}={final_states[i]}"
            for i in range(amount)
        ]
        
        return formatted_strings


