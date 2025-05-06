import numpy as np
import pandas as pd


# PROBLEM 1 - PART 1
def gibbs(A, s, t, w, burnin, its):
    n = A.shape[0]  # Total number of nodes in the graph

    # List of all nodes except the fixed source (s) and sink (t)
    other_edges = [i for i in range(n) if i != s and i != t]

    # Initialize a random binary state vector of length n
    x = np.random.randint(0, 2, n)
    x[s] = 1  # Fix source node to 1
    x[t] = 0  # Fix sink node to 0

    # Initialize marginals accumulator and list to store samples
    marginals = np.zeros(n)
    samples = []

    # Total number of Gibbs sampling iterations (burn-in + actual sampling)
    for iteration in range(burnin + its):
        # Update each node (excluding s and t) one at a time
        for other_edge in other_edges:
            neighbors = np.where(A[other_edge] != 0)[0]  # Get neighbors of node other_edge
            prob_x_1 = 1.0  # Unnormalized probability for x[other_edge] = 1
            prob_x_0 = 1.0  # Unnormalized probability for x[other_edge] = 0

            # Compute conditional probabilities based on neighbors
            for neighbor in neighbors:
                if neighbor == other_edge:
                    continue  # Skip self (shouldn't be in neighbors)

                # If neighbor is off (0), it favors x[other_edge] = 1
                if x[neighbor] == 0:
                    prob_x_1 *= np.exp(-w[other_edge, neighbor])

                # If neighbor is on (1), it favors x[other_edge] = 0
                if x[neighbor] == 1:
                    prob_x_0 *= np.exp(-w[other_edge, neighbor])

            # Normalize the probabilities
            total = prob_x_1 + prob_x_0
            if total == 0:
                prob = 0.5  # If both are zero, assign uniform probability
            else:
                prob = prob_x_1 / total  # Probability of setting x[other_edge] = 1

            # Sample x[other_edge] based on computed probability
            x[other_edge] = 1 if np.random.rand() < prob else 0

        # After burn-in period, record sample
        if iteration >= burnin:
            marginals += x  # Accumulate marginal probabilities
            samples.append(x.copy())  # Store a copy of the current state

    # Average marginals over number of samples collected
    marginals /= its

    # Return marginals and transpose of samples (n x its)
    return marginals, np.array(samples).T


# PROBLEM 1 - PART 2
if __name__ == "__main__":
    # Define a graph with 8 nodes labeled:
    # s(0), a(1), b(2), c(3), d(4), e(5), f(6), t(7)
    n = 8  # Total number of nodes
    A = np.zeros((n, n), dtype=int)  # Initialize the adjacency matrix of shape (n x n) with zeros

    # Define the edges of the graph as tuples (i, j) indicating a directed edge from i to j
    # (Note: edges are explicitly defined in both directions to make the graph undirected)
    edges = [
        (0, 1), (0, 2), (0, 3),  # Node s connected to a, b, c
        (1, 0), (1, 3), (1, 4),  # Node a connected to s, c, d
        (2, 0), (2, 3),  # Node b connected to s, c
        (3, 0), (3, 1), (3, 2), (3, 5),  # Node c connected to s, a, b, e
        (4, 1), (4, 6), (4, 7),  # Node d connected to a, f, t
        (5, 3), (5, 6), (5, 7),  # Node e connected to c, f, t
        (6, 4), (6, 5), (6, 7),  # Node f connected to d, e, t
        (7, 4), (7, 5), (7, 6)  # Node t connected to d, e, f
    ]

    # Populate the adjacency matrix using the edge list
    for i, j in edges:
        A[i, j] = 1  # Mark edge presence in the adjacency matrix

    # Initialize a weight matrix where all present edges are assigned a uniform weight of 0.5
    w = np.zeros((n, n))  # Initialize weights to zero
    w[A == 1] = 0.5  # Set weight of 0.5 for all edges present in adjacency matrix

    # Define parameters to test: burn-in steps and number of iterations after burn-in
    # These are powers of two, increasing the computational cost gradually
    params = [2 ** 6, 2 ** 10, 2 ** 14, 2 ** 18]  # = [64, 1024, 16384, 262144]

    # Create a results table with burnin as rows and iterations as columns
    table = pd.DataFrame(index=params, columns=params, dtype=float)

    # Iterate over all combinations of burn-in and sampling iterations
    for burnin in params:
        for its in params:
            # Run Gibbs sampling on the graph using given burn-in and iteration count
            # s = 0 (source node), t = 7 (target node)
            marginals, _ = gibbs(A, s=0, t=7, w=w, burnin=burnin, its=its)
            # Record the marginal probability of node e (index 5) being 1
            # Round the result to 4 decimal places for clean display
            table.loc[burnin, its] = np.round(marginals[5], 4)

    print("\nEstimated Marginal of Node e:")
    print(table.to_markdown(floatfmt=".4f"))
