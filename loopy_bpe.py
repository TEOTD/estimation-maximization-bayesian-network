import math
from collections import defaultdict as dd


def getXassign(M, w):
    x_assign = [0, 1]
    # make sure w is symmetric
    for i in range(len(w)):
        for j in range(len(w[0])):
            if M[i][j] != M[j][i]:
                print("Assymetric m")
                break
            if w[i][j] != w[j][i]:
                print("Assymetric w")
                break
    return x_assign


# returns all cliques that we care about (aka edges)
def return_cliques(M):
    cliques = []
    for i in range(len(M)):
        for j in range(len(M[0])):
            # i<j to avoid putting the same edge twice
            if M[i][j] > 0 and i < j:
                cliques.append((i, j))
    return cliques


def sumprod(M, s, t, w, its, fixed_vars=None):
    # phi and psi functions

    if fixed_vars is None:
        fixed_vars = {}

    def phi(i, x_i):
        # Priority: fixed_vars > s/t constraints
        if i in fixed_vars:
            return 1.0 if x_i == fixed_vars[i] else 0.0  # Enforce fixed values
        if i == s:
            return x_i  # s must be 1
        if i == t:
            return 1 - x_i  # t must be 0
        return 1.0  # Unfixed nodes have neutral potential

    # edge weights, determined by w[i][j]
    def psi(i, j, x_i, x_j):
        if x_i == x_j:
            return 1
        else:
            return w[i][j]

    # message passing: 
    # m_curr[(C, i, x_i)] is the message from C->i with value x_i
    # m_curr[(i, C, x_i)] is the message from i->C with value x_i

    # iterations are implemented by using and updating dicts
    m_prev = dd(lambda: 1)
    m_curr = dd(int)
    x_assign = getXassign(M, w)

    cliques = return_cliques(M)

    # the hw says time starts at t=1, so loop its-1 times
    for it in range(its - 1):

        # update m^t_{C->i}
        for C in cliques:
            for i in C:
                k = C[0] if i == C[1] else C[1]
                for x_i in x_assign:
                    m_curr[(C, i, x_i)] = 0
                    for x_k in x_assign:
                        # sum of all assignments to C 
                        m_curr[(C, i, x_i)] += psi(i, k, x_i, x_k) * m_prev[(k, C, x_k)]

                        # update m^t_{i->C}
        for C in cliques:
            for i in C:
                for x_i in x_assign:
                    val = phi(i, x_i)
                    # consider all other cliques with i in it
                    for C2 in cliques:
                        if C != C2 and i in C2:
                            val *= m_prev[(C2, i, x_i)]
                    m_curr[(i, C, x_i)] = val

        # scale values down
        maxval = max(map(abs, m_curr.values()))
        if maxval > 0:
            for i in m_curr:
                # clip all values if too low 
                m_curr[i] = max(1e-50, m_curr[i] / maxval)
        m_prev = m_curr

        # calculate beliefs
        b = dd(int)
        totals_v = dd(int)
        totals_C = dd(int)
        for i in range(len(M)):
            for x_i in x_assign:
                b[(i, x_i)] = phi(i, x_i)
                for C in cliques:
                    if i in C:
                        b[(i, x_i)] *= m_curr[(C, i, x_i)]
                totals_v[i] += b[(i, x_i)]

        # normalize beliefs
        for (i, x_i) in b:
            assert totals_v[i] != 0
            b[(i, x_i)] /= totals_v[i]

        # final belief calculation
        for C in cliques:
            for x_0 in x_assign:
                for x_1 in x_assign:
                    b[(C, (x_0, x_1))] = psi(C[0], C[1], x_0, x_1)

                    b[(C, (x_0, x_1))] *= m_curr[(C[0], C, x_0)]
                    b[(C, (x_0, x_1))] *= m_curr[(C[1], C, x_1)]

                    totals_C[C] += b[(C, (x_0, x_1))]

        for (C, x) in b:
            if type(C) == tuple:
                assert totals_C[C] != 0
                b[(C, x)] /= totals_C[C]

    # calculate bethe free energy; formula is from either slides or wikipedia
    bethe = 0

    eps = 1e-7
    for i in range(len(M)):
        for j in x_assign:
            if b[(i, j)] > eps:
                bethe -= b[(i, j)] * math.log(b[(i, j)])

    for C in cliques:
        for x_0 in x_assign:
            for x_1 in x_assign:
                prod = b[(C[0], x_0)] * b[(C[1], x_1)]
                if b[(C, (x_0, x_1))] > eps:
                    bethe -= b[(C, (x_0, x_1))] * math.log(b[(C, (x_0, x_1))] / prod)
                assert psi(C[0], C[1], x_0, x_1) != 0
                bethe += b[(C, (x_0, x_1))] * math.log(psi(C[0], C[1], x_0, x_1))
    # return partition function, beliefs

    return math.exp(bethe), b


# exact same thing as sumprod, minus the specific formula adjustments
# as well as the lack of BFE calculation
def maxprod(M, s, t, w, its):
    def phi(i, x_i):
        if i == s:
            return x_i
        if i == t:
            return 1 - x_i
        return 1

    def psi(i, j, x_i, x_j):
        if x_i == x_j:
            return 1
        else:
            return w[i][j]

    m_prev = dd(lambda: 1)
    m_curr = dd(int)

    cliques = return_cliques(M)
    x_assign = getXassign(M, w)

    for it in range(its - 1):
        for C in cliques:
            for i in C:
                k = C[0] if i == C[1] else C[1]
                for x_i in x_assign:
                    m_curr[(C, i, x_i)] = 0
                    for x_k in x_assign:
                        # sum of all assignments to C 
                        m_curr[(C, i, x_i)] = max(m_curr[(C, i, x_i)], psi(i, k, x_i, x_k) * m_prev[(k, C, x_k)])

        for C in cliques:
            for i in C:
                for x_i in x_assign:
                    val = phi(i, x_i)
                    # consider all other cliques with i in it
                    for C2 in cliques:
                        if C != C2 and i in C2:
                            val *= m_prev[(C2, i, x_i)]
                    m_curr[(i, C, x_i)] = val

        # scale values down
        maxval = max(map(abs, m_curr.values()))
        if maxval > 0:
            for i in m_curr:
                # clip all values too low 
                m_curr[i] = max(1e-20, m_curr[i] / maxval)
        m_prev = m_curr

    # calculate beliefs
    b = dd(int)
    totals_v = dd(int)
    for i in range(len(M)):
        for x_i in x_assign:
            b[(i, x_i)] = phi(i, x_i)
            for C in cliques:
                if i in C:
                    b[(i, x_i)] *= m_curr[(C, i, x_i)]
            totals_v[i] += b[(i, x_i)]

            # normalize beliefs - this is necessary for the eps calculation
    for (i, x_i) in b:
        assert totals_v[i] != 0
        b[(i, x_i)] /= totals_v[i]

    # return argmax for all values
    eps = 1e-20
    assignments = []
    for i in range(len(M)):
        if abs(b[(i, 0)] - b[(i, 1)]) < eps:
            assignments.append(0.5)
        elif b[(i, 0)] > b[(i, 1)]:
            assignments.append(0)
        else:
            assignments.append(1)

    return assignments
