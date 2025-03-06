import numpy as np

class EvenPairs:
    """
    Even Pairs (R): Given a binary sequence, compute if the number of 'ab' and 'ba' substrings is even.
    
    Example:
        Input: "aabba"
        - 'ab' appears once, 'ba' appears once.
        - Total count = 2 (even), so output is 'b'.
    
    This task is equivalent to checking if the first and last characters are the same.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initializes the class with a fixed random seed to ensure reproducibility.
        
        Args:
            seed (int): The seed for random number generation.
        """
        self.rng = np.random.default_rng(seed)

    def compute_label(self, input_string: str) -> str:
        """
        Computes whether the number of 'ab' and 'ba' substrings is even.
        
        Args:
            input_string (str): A binary string (composed of 'a' and 'b').
        
        Returns:
            str: 'b' if even, 'a' if odd.
        """
        count = sum(1 for i in range(len(input_string) - 1) 
                    if input_string[i:i+2] in ('ab', 'ba'))
        return 'b' if count % 2 == 0 else 'a'

    def sample_batch(self, amount: int, length: int) -> list:
        """
        Generates a batch of random binary strings and their corresponding labels.
        
        Args:
            amount (int): Number of strings to generate.
            length (int): Length of each string.
        
        Returns:
            list: Each element is a string of format "<input_string>=<label>"
        """
        samples = [
            ''.join(self.rng.choice(['a', 'b'], size=length)) for _ in range(amount)
        ]
        
        formatted_strings = [
            f"{sample}{self.compute_label(sample)}" for sample in samples
        ]
        
        return formatted_strings
    
if __name__ == "__main__":
    # Example usage
    task = EvenPairs()
    print(task.sample_batch(amount=5, length=5))
