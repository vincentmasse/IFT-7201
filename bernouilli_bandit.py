from scipy.stats import bernoulli
import numpy as np

class BernoulliBandit:
    def __init__(self, k, rand_seed = 100):
        np.random.seed(rand_seed)
        self.mu = np.random.random(k)
        self.k_star = max(self.mu)
        self.cumul_regret = 0


    def play(self, k):
        self.cumul_regret += self.k_star - self.mu[k]
        return bernoulli.rvs(self.mu[k])

    def get_K(self):
        return len(self.mu)

    def get_cumul_regret(self):
        return self.cumul_regret
