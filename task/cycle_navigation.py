import numpy as np

class CycleNavigation:
    """A task with the goal of computing the final state on a circle.

    The input is a sequence of actions, composed of -1s, 0s, or 1s. 
    The actions indicate movements on a finite-length circle:
        - 0 means stay,
        - 1 means move right,
        - -1 means move left.
    
    The agent starts at position 0, and the goal is to compute the final position 
    on the circle after executing all actions.

    By default, the length of the circle is 5.
    """

    def __init__(self, cycle_length=5):
        self.cycle_length = cycle_length

    def sample_batch(self, batch_size: int, length: int) -> list:
        """Generates a batch of action sequences and their corresponding final positions.

        Args:
            batch_size (int): Number of sequences in the batch.
            length (int): Length of each action sequence.

        Returns:
            list: Each element is a string of format "<actions>=<binary_output>"
        """
        rng = np.random.default_rng()
        actions = rng.choice([-1, 0, 1], size=(batch_size, length))

        # Compute final states
        final_states = np.sum(actions, axis=1) % self.cycle_length

        # Convert to formatted strings
        formatted_strings = []
        for i in range(batch_size):
            action_str = ''.join(map(str, actions[i]))  # Convert actions to string
            binary_output = format(1 << final_states[i], f'0{self.cycle_length}b')  # One-hot binary
            formatted_strings.append(f"{action_str}={binary_output}")

        return formatted_strings

# Ejemplo de uso
task = CycleNavigation()
batch = task.sample_batch(batch_size=5, length=5)
print(batch)
print(len(batch))
print(batch[:5])
batch = task.sample_batch(batch_size=5, length=5)
print(len(batch))
print(batch[:5])


