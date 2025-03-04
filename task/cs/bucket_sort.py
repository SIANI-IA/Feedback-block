import numpy as np

class BucketSort:
    """
    Bucket Sort (CS): Given a string over an alphabet of fixed size (5 in our case),
    return the sorted string. Since the alphabet has a fixed size, the task can be
    solved via bucket sort, which only requires a finite amount of counters (5 counters).
    
    Example:
        Input: "421302214"
        Output: "011222344"
    """
    
    def __init__(self, alphabet_size: int = 5, seed: int = 42):
        self.alphabet_size = alphabet_size
        self.rng = np.random.default_rng(seed)

    def sort_string(self, input_string: str) -> str:
        """
        Sorts the input string using bucket sort.
        
        Args:
            input_string (str): A string composed of digits from 0 to alphabet_size - 1.
        
        Returns:
            str: The sorted string.
        """
        # Count occurrences of each digit
        counts = np.zeros(self.alphabet_size, dtype=int)
        for char in input_string:
            counts[int(char)] += 1
        
        # Reconstruct sorted string
        sorted_string = ''.join(str(i) * counts[i] for i in range(self.alphabet_size))
        return sorted_string

    def sample_batch(self, amount: int, length: int) -> list:
        """
        Generates a batch of random strings and their corresponding sorted outputs.
        
        Args:
            amount (int): Number of strings to generate.
            length (int): Length of each string.
        
        Returns:
            list: Each element is a string of format "<input_string>=<sorted_string>"
        """
        samples = self.rng.integers(0, self.alphabet_size, size=(amount, length))
        
        formatted_strings = [
            f"{''.join(map(str, sample))}{self.sort_string(''.join(map(str, sample)))}"
            for sample in samples
        ]
        
        return formatted_strings
    
if __name__ == "__main__":
    # Example usage
    task = BucketSort()
    print(task.sample_batch(amount=5, length=10))
