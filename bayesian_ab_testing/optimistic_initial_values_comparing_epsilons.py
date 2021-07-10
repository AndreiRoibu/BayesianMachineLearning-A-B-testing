import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from bayesian_ab_testing.epsilon_greedy_comparing_epsilons import experiment as epsilon_experiment_comparison

class Bandit:
    def __init__(self, m, upper_limit):
        self.m = m
        self.mean = upper_limit
        self.N = 1.0
    
    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1.0 - 1.0/self.N) * self.mean + 1.0 / self.N * x

def run_experiment(m1, m2, m3, N, upper_limit=10):
    bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]
    data = np.empty(N)

    for i in range(N):
        # optimistic initial values used to select the next bandit
        j = np.argmax([bandit.mean for bandit in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1.0)

    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)

    return cumulative_average

if __name__ == '__main__':
    c_1 = epsilon_experiment_comparison(1.0, 2.0, 3.0, 0.1, 100000)
    oiv = run_experiment(1.0, 2.0, 3.0, 100000)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(oiv, label='optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()


    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(oiv, label='optimistic')
    plt.legend()
    plt.show()