import numpy as np
from scipy.optimize import minimize, Bounds
from bernouilli_bandit import BernoulliBandit


def d(p, q):
    def part(p1, q1):
        if p1 == 0:
            ans = 0
        elif q1 == 0:
            ans = np.inf
        else:
            ans = p1 * np.log(p1 / q1)
        return ans

    return part(p, q) + part(1 - p, 1 - q)


q_bounds = (0, 1)


def ucb_opt(n, s, t, c):
    def to_optimize(q):
        return np.power(np.log(t) + c * np.log(np.log(t+1)) - n * d(s / n, q), 2)

    q = minimize(to_optimize, 1, bounds=((s / n, max(s / n, 1 - 1e-16)),))
    return q.x[0]

def ucb_gaussian(n, s, t, c):
    return s / n + np.sqrt((1 / 2) * (np.log(t + 1) + c * np.log(np.log(t + 1))) / n)




def kl_ucb(bandit, T, ucb, c=3):
    K = bandit.get_K()
    n, s = np.zeros(K), np.zeros(K)
    for t in range(0, T):
        if t < k:
            k_t = t
        else:
            ucb_t = [ucb(n[i], s[i], t, c) for i in range(0, k)]
            index = np.array([i for i in range(k)])
            index_max = (ucb_t == max(ucb_t))
            possible_ind = index[index_max]
            k_t = possible_ind[np.argmin(n[possible_ind])]
        s[k_t] += bandit.play(k_t)
        n[k_t] += 1

def kl_ucb_bernoulli(bandit, T, c=3):
    kl_ucb(bandit, T, ucb_opt)

def kl_ucb_gaussian(bandit, T, c=3):
    kl_ucb(bandit, T, ucb_gaussian)

if __name__ == "__main__":
    k = 10
    T = 1000
    c = 3

    bandit1 = BernoulliBandit(k)
    kl_ucb_bernoulli(bandit1, T)

    bandit2 = BernoulliBandit(k)
    kl_ucb_gaussian(bandit2, T)

    print(bandit1.get_cumul_regret())
    print(bandit2.get_cumul_regret())
