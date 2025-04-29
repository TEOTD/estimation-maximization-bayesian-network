import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict

# Load and preprocess data
data = pd.read_csv('house-votes-84.data', header=None)
data = data.iloc[:300, :4]  # First 300 rows, columns 0 (party), 1-3 (votes)


# Convert data to numerical values (0/1/NaN)
def preprocess(val):
    if val == 'democrat': return 0
    if val == 'republican': return 1
    if val == 'y': return 1
    if val == 'n': return 0
    return np.nan


data = data.applymap(preprocess)
train_data = data.iloc[:250, :3].values  # First three features: party, vote1, vote2
test_data = data.iloc[250:300, :3].values  # Remaining for testing

# Define DAG structures (example of 8 possible DAGs)
dags = [
    # DAG 0: Party -> Vote1 -> Vote2
    {0: [], 1: [0], 2: [1]},
    # DAG 1: Party -> Vote2 -> Vote1
    {0: [], 2: [0], 1: [2]},
    # DAG 2: Vote1 -> Party -> Vote2
    {1: [], 0: [1], 2: [0]},
    # DAG 3: Vote1 -> Vote2 -> Party
    {1: [], 2: [1], 0: [2]},
    # DAG 4: Vote2 -> Party -> Vote1
    {2: [], 0: [2], 1: [0]},
    # DAG 5: Vote2 -> Vote1 -> Party
    {2: [], 1: [2], 0: [1]},
    # DAG 6: Party <- Vote1 -> Vote2
    {1: [], 0: [1], 2: [1]},
    # DAG 7: Party -> Vote1 <- Vote2
    {0: [], 1: [0, 2], 2: []},
]


def learn_bn(dag, data, max_iter=50, epsilon=1e-3):
    n_vars = data.shape[1]
    cpts = {}
    parents = {var: dag[var] for var in dag}

    # Initialize CPTs randomly
    for var in range(n_vars):
        parent_vars = parents[var]
        if not parent_vars:
            prob = np.random.dirichlet([1] * 2)
            cpts[var] = prob
        else:
            parent_states = [2] * len(parent_vars)  # Binary parents
            cpt = np.random.dirichlet([1] * 2, size=parent_states)
            cpts[var] = cpt

    prev_log_likelihood = -np.inf
    for _ in range(max_iter):
        # E-step: Compute expected counts
        counts = defaultdict(lambda: defaultdict(float))
        for row in data:
            # Compute posterior for missing variables
            observed = {i: row[i] for i in range(n_vars) if not np.isnan(row[i])}
            hidden = [i for i in range(n_vars) if np.isnan(row[i])]

            # Generate all possible states for hidden variables
            hidden_states = list(product([0, 1], repeat=len(hidden)))
            total_prob = 0.0
            for state in hidden_states:
                full_state = observed.copy()
                for i, s in zip(hidden, state):
                    full_state[i] = s

                # Compute joint probability
                prob = 1.0
                for var in range(n_vars):
                    if var in parents[var]:
                        parent_vals = tuple(full_state[p] for p in parents[var])
                        prob *= cpts[var][parent_vals][full_state[var]]
                    else:
                        prob *= cpts[var][full_state[var]]
                total_prob += prob
                # Update counts
                for var in range(n_vars):
                    if var in parents[var]:
                        parent_vals = tuple(full_state[p] for p in parents[var])
                        counts[var][(parent_vals, full_state[var])] += prob
                    else:
                        counts[var][(full_state[var],)] += prob

            # Normalize counts
            for var in counts:
                for key in counts[var]:
                    counts[var][key] /= total_prob

        # M-step: Update CPTs
        new_cpts = {}
        for var in range(n_vars):
            parent_vars = parents[var]
            if not parent_vars:
                total = counts[var].get((0,), 0) + counts[var].get((1,), 0)
                prob0 = counts[var].get((0,), 0) / total if total > 0 else 0.5
                new_cpts[var] = np.array([prob0, 1 - prob0])
            else:
                parent_states = list(product([0, 1], repeat=len(parent_vars)))
                cpt = np.zeros((len(parent_states), 2))
                for i, p_state in enumerate(parent_states):
                    total = counts[var].get((p_state, 0), 0) + counts[var].get((p_state, 1), 0)
                    if total == 0:
                        cpt[i] = [0.5, 0.5]
                    else:
                        cpt[i, 0] = counts[var].get((p_state, 0), 0) / total
                        cpt[i, 1] = counts[var].get((p_state, 1), 0) / total
                new_cpts[var] = cpt

        # Check convergence
        log_likelihood = 0.0
        for row in data:
            observed = {i: row[i] for i in range(n_vars) if not np.isnan(row[i])}
            prob = 0.0
            for var in range(n_vars):
                if var in observed:
                    parent_vals = tuple(observed[p] for p in parents[var])
                    prob += np.log(new_cpts[var][parent_vals][observed[var]])
            log_likelihood += prob

        if abs(log_likelihood - prev_log_likelihood) < epsilon:
            break
        prev_log_likelihood = log_likelihood
        cpts = new_cpts

    return cpts


def predict_party(cpts, dag, test_row):
    # test_row: [party (NaN), vote1, vote2]
    observed = {1: test_row[1], 2: test_row[2]}  # vote1 and vote2 are observed
    hidden = 0  # party is to be predicted

    prob0 = 1.0  # P(party=0 | vote1, vote2)
    prob1 = 1.0  # P(party=1 | vote1, vote2)

    # Compute using BN structure
    for var in [0, 1, 2]:
        parents = dag[var]
        if var == hidden:
            # Sum over possible values of party
            pass  # Handled below
        else:
            if var in observed:
                val = observed[var]
                if parents:
                    parent_vals = tuple(observed[p] for p in parents)
                    p = cpts[var][parent_vals][val]
                else:
                    p = cpts[var][val]
                prob0 *= p if var != hidden else 1
                prob1 *= p if var != hidden else 1

    # Normalize
    total = prob0 + prob1
    return 0 if prob0 > prob1 else 1


# Train and evaluate each DAG
results = []
for dag_idx, dag in enumerate(dags):
    # Learn CPTs using EM
    cpts = learn_bn(dag, train_data)

    # Predict on test data
    correct = 0
    for row in test_data:
        true_party = row[0]
        if np.isnan(true_party):
            continue  # Skip if party is missing
        pred_party = predict_party(cpts, dag, row)
        correct += (pred_party == true_party)

    accuracy = correct / len(test_data)
    results.append((dag_idx, accuracy))

# Display results
print("Model\tAccuracy")
for res in results:
    print(f"{res[0]}\t{res[1]:.2f}")

# Part 2: Check if different EM runs produce different models
# (Run learn_bn multiple times with different initializations and compare CPTs)

# Part 3: Evaluate dependency on model
# Compare accuracies across models to see if they vary significantly