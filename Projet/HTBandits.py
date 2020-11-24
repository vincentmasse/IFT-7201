'''Dans le cas simple de bras à distributions de Cauchy,
    il semble approprié que la mesure de performance (regret)
    pourrait simplement être défini en fonction du paramètre
    de localisation de la loi au lieu de la moyenne (qui n'existe pas).
    La classe suivante définit un bandit à distributions de Cauchy avec cette convention pour cumuler le regret.
'''

import numpy as np
import scipy

class CauchyBandit:
    def __init__(self, loc, scale, seed=None):
        ''' Entrées:
                    - loc: un array décrivant les paramètres de localisations des lois de Cauchy décrivant les actions
                    - scale: un array décrivant les paramètres d'échelle des lois de Cauchy décrivant les actions
                    - seed (optionnel): un seed. '''

        np.random.seed(seed=seed)

        self.loc = loc

        self.scale = scale

        self.ps_regrets = []

        self.locstar = max(self.loc)

        self.kstar = np.argmax(loc)

        self.gaps = self.locstar - loc

    def get_K(self):
        ''' Return the number of actions . '''

        K = len(self.loc)

        return K

    def play(self, action):
        ''' Accept a parameter 0 <= k < K, logs the instant pseudo - regret ,
        and return the realization of a Bernoulli random variable with P(X =1)
        being the mean of the given action . '''

        ps_regrets_inst = self.loc - self.loc[action]

        # méthode de conténation append
        self.ps_regrets.append(self.gaps[action])

        reward = scipy.stats.cauchy.rvs(self.loc[action], self.scale[action], size=1)

        return reward

    def get_cumulative_regret(self):
        ''' Return an array of the cumulative sum of pseudo - regret per round . '''
        return np.cumsum(self.ps_regrets)


# Classe ParetoBandit avec mesure de performance basée sur le gap défini comme l'écart entre le paramètre $x_m_star$
# et le paramètre $x_m$ du bras choisi.
class ParetoBandit:
    def __init__(self, x_m, k, seed=None):
        ''' Entrées:
                    - x_m: un array décrivant les paramètres x_m de localisations des lois de Pareto décrivant les actions
                    - k: un array décrivant les paramètres k des lois de Pareto décrivant les actions
                    - seed (optionnel): un seed. '''

        np.random.seed(seed=seed)

        self.x_m = x_m

        self.k = k

        self.ps_regrets = []

        self.x_m_star = max(self.x_m)

        self.kstar = np.argmax(x_m)

        self.gaps = self.x_m_star - x_m

    def get_K(self):
        ''' Return the number of actions . '''

        K = len(self.x_m)

        return K

    def play(self, action):
        ''' Accept an 0<=action<=(K-1), logs the instant pseudo - regret ,
        and return the realization of a Pareto distribution with the parameters
        x_m and k of the given action. '''

        # méthode de conténation append
        self.ps_regrets.append(self.gaps[action])

        reward = scipy.stats.pareto.rvs(b=self.k[action], loc=0, scale=self.x_m[action], size=1)

        return reward

    def get_cumulative_regret(self):
        ''' Return an array of the cumulative sum of pseudo - regret per round . '''
        return np.cumsum(self.ps_regrets)
