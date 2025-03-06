import numpy as np

class ModularArithmetic:
    """
    Modular Arithmetic (Simple) (R): Given a sequence of numbers in {0, 1, 2, 3, 4}
    and operations in {+, -, *}, compute the result modulo 5.
    
    Example:
        Input: "1+2-4"
        - Computation: (1 + 2 - 4) % 5 = 4
        - Output: "4"
    
    The input sequence must be of odd length, so if an even length is sampled,
    it is adjusted by adding an additional random operation and number.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initializes the class with a fixed random seed to ensure reproducibility.
        
        Args:
            seed (int): The seed for random number generation.
        """
        self.rng = np.random.default_rng(seed)
        self.operators = ['+', '-', '*']
        self.numbers = ['0', '1', '2', '3', '4']
    
    def compute_result(self, input_string: str) -> str:
        """
        Computes the result of the expression modulo 5.
        
        Args:
            input_string (str): A mathematical expression as a string.
        
        Returns:
            str: The computed result modulo 5.
        """
        try:
            result = eval(input_string) % 5
        except Exception:
            result = 0  # Fallback in case of malformed input
        return str(result)
    
    def sample_batch(self, amount: int, length: int) -> list:
        """
        Generates a batch of random arithmetic expressions and their results modulo 5.
        
        Args:
            amount (int): Number of expressions to generate.
            length (int): Length of each expression (must be odd).
        
        Returns:
            list: Each element is a string of format "<expression>=<result>"
        """
        if length % 2 == 0:
            length += 1  # Ensure odd length
        
        samples = []
        for _ in range(amount):
            expr = [self.rng.choice(self.numbers)]
            for _ in range((length - 1) // 2):
                expr.append(self.rng.choice(self.operators))
                expr.append(self.rng.choice(self.numbers))
            expression = ''.join(expr)
            samples.append(f"{expression}={self.compute_result(expression)}")
        
        return samples
    
if __name__ == "__main__":
    # Example usage
    task = ModularArithmetic()
    print(task.sample_batch(amount=5, length=5))