import numpy as np

def simulate_angles(num_samples=100000):
    """
    Simulate the angles and calculate the probability that |pos_neg| > |pos_neutral| and |pos_neg| > |neg_neutral|.

    Parameters:
        num_samples (int): Number of random samples to generate.

    Returns:
        tuple: Probability that |pos_neg| > |pos_neutral| and |pos_neg| > |neg_neutral|, 
               and average angle differences.
    """
    # Initialize counters and lists
    conditions_met_count = 0
    pos_neg_differences = []
    pos_neutral_differences = []
    neg_neutral_differences = []
    
    for _ in range(num_samples):
        # Generate random values in the range [0, π/2]
        pos = np.random.uniform(0, np.pi/2)
        neutral = np.random.uniform(0, np.pi/2)
        neg = np.random.uniform(0, np.pi/2)

        # Calculate absolute differences
        pos_neg = np.abs(pos - neg)
        pos_neutral = np.abs(pos - neutral)
        neg_neutral = np.abs(neg - neutral)

        # Convert angle differences to their sine equivalents (using 1 - cos)
        pos_neg_degree = 1 - np.cos(pos_neg)
        pos_neutral_degree = 1 - np.cos(pos_neutral)
        neg_neutral_degree = 1 - np.cos(neg_neutral)

        # Store the differences (use degree representation)
        pos_neg_differences.append(pos_neg_degree)
        pos_neutral_differences.append(pos_neutral_degree)
        neg_neutral_differences.append(neg_neutral_degree)

        # Check the conditions
        if pos_neg_degree > pos_neutral_degree and pos_neg_degree > neg_neutral_degree:
            conditions_met_count += 1

    # Calculate the probability
    probability = conditions_met_count / num_samples

    # Calculate average angle differences (in terms of degree representation)
    avg_pos_neg = np.mean(pos_neg_differences)
    avg_pos_neutral = np.mean(pos_neutral_differences)
    avg_neg_neutral = np.mean(neg_neutral_differences)

    return probability, avg_pos_neg, avg_pos_neutral, avg_neg_neutral

def main():
    # Set the number of samples
    num_samples = 100000

    # Run the simulation
    probability_result, avg_pos_neg, avg_pos_neutral, avg_neg_neutral = simulate_angles(num_samples)
    
    # Print the results
    print(f"The probability that |pos_neg| > |pos_neutral| and |pos_neg| > |neg_neutral| is: {probability_result:.4f}")
    print(f"Average pos_neg difference: {avg_pos_neg:.4f}")
    print(f"Average pos_neutral difference: {avg_pos_neutral:.4f}")
    print(f"Average neg_neutral difference: {avg_neg_neutral:.4f}")

if __name__ == "__main__":
    main()
