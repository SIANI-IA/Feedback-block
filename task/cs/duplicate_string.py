import numpy as np

class DuplicateString:
    """
    Duplicate String (CS): Given a binary string, output the string twice.
    
    Example:
        Input: "abaab"
        Output: "abaababaab"
    
    This task is CS since it corresponds to the well-known language {ww | w is a word}.
    """
    
    def __init__(self):
        pass

    def duplicate(self, input_string: str) -> str:
        """
        Duplicates the input string.
        
        Args:
            input_string (str): A binary string (composed of 'a' and 'b').
        
        Returns:
            str: The duplicated string.
        """
        return input_string + input_string

    def sample_batch(self, amount: int, length: int) -> list:
        """
        Generates a batch of random binary strings and their corresponding duplicated outputs.
        
        Args:
            amount (int): Number of strings to generate.
            length (int): Length of each string.
        
        Returns:
            list: Each element is a string of format "<input_string>=<duplicated_string>"
        """
        rng = np.random.default_rng()
        samples = [
            ''.join(rng.choice(['a', 'b'], size=length)) for _ in range(amount)
        ]
        
        formatted_strings = [
            f"{self.duplicate(sample)}" for sample in samples
        ]
        
        return formatted_strings
    
if __name__ == "__main__":
    # Example usage
    task = DuplicateString()
    print(task.sample_batch(amount=5, length=5))
