import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Bandit:
    def __init__(self, m):
        self.m = m
        self.m_estimate = 0.0
        self.N = 0.0 # number of samples so far
    
    def pull(self):
        return np.random.randn() + self.m # Gaussian/Normal distribution

    def update(self, x):
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N

def experiment(m1, m2, m3, eps, N):
    # ms represent the means of each bandit
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in range(N):

        # use epsilon-greedy to select the next bandit
        # either randomly selecting one of the bandits
        # or selecting the best one based on p_estimate

        p = np.random.random()
        if p < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([bandit.m_estimate for bandit in bandits])
         
        # pull the arm of the bandit with the largest sample
        x = bandits[j].pull()

        # update the distriobution forthe bandit whose arm we pooled
        bandits[j].update(x)

        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plot the results
    plt.figure()
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1, label='bandit 1')
    plt.plot(np.ones(N)*m2, label='bandit 2')
    plt.plot(np.ones(N)*m3, label='bandit 3')
    plt.xscale('log')
    plt.legend()
    plt.show()

    for b in bandits:
        print(b.m_estimate)

    return cumulative_average

if __name__ == '__main__':
    c_1 = experiment(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = experiment(1.0, 2.0, 3.0, 0.05, 100000)
    c_01 = experiment(1.0, 2.0, 3.0, 0.01, 100000)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()