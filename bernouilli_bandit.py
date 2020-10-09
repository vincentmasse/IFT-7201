from scipy.stats import bernoulli
import numpy as np

class BernoulliBandit:
    def __init__(self, k, rand_seed = None):
        np.random.seed(rand_seed)
        self.mu = np.random.random(k)
        self.k_star = max(self.mu)
        self.regret = []


    def play(self, k):
        self.regret.append(self.k_star - self.mu[k])
        return bernoulli.rvs(self.mu[k])

    def get_K(self):
        return len(self.mu)

    def get_regret(self):
        return self.regret

    def get_cumul_regret(self):
        cumul_regret = np.cumsum(self.regret)
        return cumul_regret
