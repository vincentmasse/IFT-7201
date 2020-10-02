import numpy as np
from scipy.optimize import minimize, Bounds


def d(p, q):
    if q != 0:
        diverge = p * np.ln(p / q) + (1 - p) * np.ln((1 - p) / (1 - q))
    elif p == 0:
        diverge = 0
    else:
        diverge = np.inf

    return diverge


def ucb(n, s, t, c):
    def to_optimize(q):
        return np.ln(t) + c * np.ln(np.ln(t)) - n * d(s / n, q)

    q = minimize(to_optimize, Bounds([0, 1]), to_optimize >= 0)
    return q


def reward(k):
    return np.random.bernoulli(p[k])


k = 10
N = 10e10
n = np.zeros(k)
s = np.zeros(k)
p = np.random(k)

for t in range(0, k - 1):
    n[t] += 1
    s[t] += reward(t)
for t in range(t, N - 1):
    ucb_t = [ucb(i) for i in range(0, k - 1)]
    a = np.argmax(ucb_t)
    n[a] += 1
    s[a] += reward(a)
