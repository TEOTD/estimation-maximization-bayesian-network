from collections import defaultdict

import numpy as np
import pandas as pd

from gibbs_sampling import gibbs
from loopy_bpe import sumprod


def cutsem(A, L, samples, s=0, t=None, bp_its=1000, learning_rate=1.0, max_iter=1000):
    n = A.shape[0]
    m = samples.shape[1]
    if t is None:
        t = n - 1

    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    theta = defaultdict(float)
    for edge in edges:
        theta[edge] = 1.0

    for _ in range(max_iter):
        w = np.zeros((n, n))
        for (i, j) in edges:
            w_ij = np.exp(theta[(i, j)])
            w[i][j] = w_ij
            w[j][i] = w_ij

        expectations_empirical = np.zeros_like(A, dtype=float)
        for sample_idx in range(m):
            fixed_vars = {i: samples[i, sample_idx] for i in range(n) if L[i] == 0}
            fixed_vars[s] = 1
            fixed_vars[t] = 0

            _, beliefs = sumprod(A, s, t, w, its=bp_its, fixed_vars=fixed_vars)

            for (i, j) in edges:
                if i in fixed_vars and j in fixed_vars:
                    disagreement = int(fixed_vars[i] != fixed_vars[j]) / m
                else:
                    prob_01 = beliefs.get(((i, j), (0, 1)), 0.0)
                    prob_10 = beliefs.get(((i, j), (1, 0)), 0.0)
                    disagreement = (prob_01 + prob_10) / m
                expectations_empirical[i, j] += disagreement
                expectations_empirical[j, i] += disagreement

        _, model_beliefs = sumprod(A, s, t, w, bp_its)
        expectations_model = np.zeros_like(A, dtype=float)
        for (i, j) in edges:
            prob_01 = model_beliefs.get(((i, j), (0, 1)), 0.0)
            prob_10 = model_beliefs.get(((i, j), (1, 0)), 0.0)
            expectations_model[i, j] = prob_01 + prob_10
            expectations_model[j, i] = expectations_model[i, j]

        for edge in edges:
            gradient = expectations_empirical[edge] - expectations_model[edge]
            delta = learning_rate * gradient
            theta[edge] += delta

    w = np.zeros((n, n))
    for (i, j) in edges:
        w_ij = np.exp(theta[(i, j)])
        w[i][j] = w_ij
        w[j][i] = w_ij
    return w


if __name__ == '__main__':
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    true_w = np.array([[0, np.exp(1), 0], [np.exp(1), 0, np.exp(1)], [0, np.exp(1), 0]])
    s, t = 0, 2

    L = np.array([0, 1, 0])

    sample_sizes = [10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    results = []

    z_true, _ = sumprod(A, s, t, true_w, its=1000)

    for m in sample_sizes:
        _, samples = gibbs(A, s=s, t=t, w=true_w, burnin=1000, its=m)
        latent_nodes = np.where(L == 1)[0]
        samples[latent_nodes, :] = 999

        learned_w = cutsem(A, L, samples, bp_its=1000, max_iter=1000)
        z_estimated, _ = sumprod(A, s, t, learned_w, its=1000)
        results.append(z_estimated)

    df = pd.DataFrame({
        "Samples": sample_sizes,
        "Estimated Z": np.round(results, 4),
        "True Z": np.round(z_true, 4)
    })

    print("\nEstimated Partition Function vs. True Z (with Latent Variables):")
    print(df.to_markdown(index=False))
