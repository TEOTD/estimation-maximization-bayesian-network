import numpy as np
from collections import defaultdict
from loopy_bpe import sumprod


def colorem(A, L, samples, s=0, t=None, bp_its=100, max_em_iter=50, learning_rate=0.1):
    n = A.shape[0]
    if t is None:
        t = n - 1  # Assume t is the last node if not specified

    edges = [(i, j) for i in range(n) for j in range(n) if A[i, j] and i < j]
    m_samples = samples.shape[1]

    # Initialize weights (theta as log-weights)
    theta = defaultdict(float)
    for edge in edges:
        theta[edge] = 0.0  # Initial log-weight (w=exp(0)=1)

    for _ in range(max_em_iter):
        # E-step: Compute empirical edge disagreements
        empirical = np.zeros_like(A, dtype=float)

        for sample_idx in range(m_samples):
            # Clamp observed variables (non-L) and s, t
            observed_vars = {s: 1, t: 0}
            for i in range(n):
                if not L[i] and i != s and i != t:
                    observed_vars[i] = samples[i, sample_idx]

            # Run BP with clamped observed variables
            _, beliefs = sumprod_clamped(A, s, t, theta_to_weights(theta, edges), its=bp_its, observed=observed_vars)

            # Accumulate empirical expectations
            for (i, j) in edges:
                prob_01 = beliefs.get(((i, j), (0, 1)), 0)
                prob_10 = beliefs.get(((i, j), (1, 0)), 0)
                empirical[i, j] += prob_01 + prob_10
                empirical[j, i] = empirical[i, j]

        empirical /= m_samples  # Average over samples

        # M-step: Compute model expectations and update weights
        # Run BP on full model (only s and t are clamped)
        _, model_beliefs = sumprod_clamped(A, s, t, theta_to_weights(theta, edges), its=bp_its, observed={s: 1, t: 0})

        model_expectations = np.zeros_like(A, dtype=float)
        for (i, j) in edges:
            prob_01 = model_beliefs.get(((i, j), (0, 1)), 0)
            prob_10 = model_beliefs.get(((i, j), (1, 0)), 0)
            model_expectations[i, j] = prob_01 + prob_10
            model_expectations[j, i] = model_expectations[i, j]

        # Update theta (log-weights) using gradient ascent
        for (i, j) in edges:
            grad = empirical[i, j] - model_expectations[i, j]
            theta[(i, j)] += learning_rate * grad

    # Convert final theta to weights
    w = np.zeros_like(A, dtype=float)
    for (i, j) in edges:
        w[i, j] = np.exp(theta[(i, j)])
        w[j, i] = w[i, j]

    return w


def sumprod_clamped(M, s, t, w, its, observed):
    """Modified sumprod to clamp observed variables."""

    def phi(i, x_i):
        if i in observed:
            return 1.0 if x_i == observed[i] else 0.0
        if i == s:
            return x_i  # Enforce s=1
        if i == t:
            return 1 - x_i  # Enforce t=0
        return 1.0

    # The rest of the sumprod implementation with the custom phi
    # (Refer to loopy_bpe.py and adjust to use the new phi)
    # This is a simplified placeholder; actual implementation requires integrating the original sumprod code.
    _, beliefs = sumprod(M, s, t, w, its)
    return 1.0, beliefs  # Placeholder return


def theta_to_weights(theta, edges):
    n = max(max(i, j) for (i, j) in edges) + 1
    w = np.zeros((n, n))
    for (i, j) in edges:
        w[i, j] = np.exp(theta[(i, j)])
        w[j, i] = w[i, j]
    return w