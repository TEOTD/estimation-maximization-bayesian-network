from collections import defaultdict

import numpy as np
import pandas as pd

from gibbs_sampling import gibbs
from loopy_bpe import sumprod


def cutsem(A, L, samples, s=0, t=None, bp_its=1000, learning_rate=1.0, max_iter=1000, tolerance=1e-5):
    n = A.shape[0]
    m = samples.shape[1]
    if t is None:
        t = n - 1

    # Extract list of edges from adjacency matrix
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]

    # Initialize log-potentials (theta) to 1.0
    theta = defaultdict(float)
    for edge in edges:
        theta[edge] = 1.0

    for _ in range(max_iter):
        max_delta = 0.0
        # Compute weight matrix w from current theta
        w = np.zeros((n, n))
        for (i, j) in edges:
            w_ij = np.exp(theta[(i, j)])
            w[i][j] = w_ij
            w[j][i] = w_ij

        # Empirical expectations based on Gibbs samples
        expectations_empirical = np.zeros_like(A, dtype=float)
        for sample_idx in range(m):
            # Fix values for observed nodes and source/target
            fixed_vars = {i: samples[i, sample_idx] for i in range(n) if L[i] == 0}
            fixed_vars[s] = 1
            fixed_vars[t] = 0

            # Run loopy belief propagation to get pairwise marginals
            _, beliefs = sumprod(A, s, t, w, its=bp_its, fixed_vars=fixed_vars)

            # Update empirical expectations for edge disagreements
            for (i, j) in edges:
                if i in fixed_vars and j in fixed_vars:
                    disagreement = int(fixed_vars[i] != fixed_vars[j]) / m
                else:
                    prob_01 = beliefs.get(((i, j), (0, 1)), 0.0)
                    prob_10 = beliefs.get(((i, j), (1, 0)), 0.0)
                    disagreement = (prob_01 + prob_10) / m
                expectations_empirical[i, j] += disagreement
                expectations_empirical[j, i] += disagreement

        # Model expectations (no conditioning)
        _, model_beliefs = sumprod(A, s, t, w, bp_its)
        expectations_model = np.zeros_like(A, dtype=float)
        for (i, j) in edges:
            prob_01 = model_beliefs.get(((i, j), (0, 1)), 0.0)
            prob_10 = model_beliefs.get(((i, j), (1, 0)), 0.0)
            expectations_model[i, j] = prob_01 + prob_10
            expectations_model[j, i] = expectations_model[i, j]

        # Gradient ascent update on theta
        for edge in edges:
            gradient = expectations_empirical[edge] - expectations_model[edge]
            delta = learning_rate * gradient
            theta[edge] += delta
            max_delta = max(max_delta, abs(float(delta)))

        if max_delta < float(tolerance):
            break

    # Final weight matrix computation from learned theta
    w = np.zeros((n, n))
    for (i, j) in edges:
        w_ij = np.exp(theta[(i, j)])
        w[i][j] = w_ij
        w[j][i] = w_ij

    return w


if __name__ == '__main__':
    # Test case remains unchanged from original
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    true_w = np.array([[0, np.exp(1), 0], [np.exp(1), 0, np.exp(1)], [0, np.exp(1), 0]])

    s, t = 0, 2
    L = np.array([0, 1, 0])

    sample_sizes = [10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    results = []

    # True partition function calculation
    z_true, _ = sumprod(A, s, t, true_w, its=1000)

    # Run experiments for increasing sample sizes
    for m in sample_sizes:
        # Generate samples with latent nodes marked
        _, samples = gibbs(A, s=s, t=t, w=true_w, burnin=1000, its=m)
        samples[np.where(L == 1)[0], :] = 999  # Mask latent nodes

        # Learn weights with improved algorithm
        learned_w = cutsem(A, L, samples)

        # Estimate partition function
        z_estimated, _ = sumprod(A, s, t, learned_w, its=1000)
        results.append(z_estimated)

    # Results comparison
    df = pd.DataFrame({
        "Samples": sample_sizes,
        "Estimated Z": np.round(results, 4),
        "True Z": np.round(z_true, 4)
    })
    print("\nPartition Function Estimation Results:")
    print(df.to_markdown(index=False))
