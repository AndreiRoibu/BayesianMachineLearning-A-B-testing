import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

NUM_TRIALS = 10000
EPSILON = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 5.0
        self.N = 1.0 # number of samples so far
    
    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.0
        self.p_estimate = ((self.N - 1.0) * self.p_estimate + x) / self.N

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        # use optimistic initial values to select the next bandit
        j = np.argmax([bandit.p_estimate for bandit in bandits])
            
        # pull the arm of the bandit with the largest sample
        x = bandits[j].pull()

        # update the rewards log
        rewards[i] = x

        # update the distriobution forthe bandit whose arm we pooled
        bandits[j].update(x)

    for i, bandit in enumerate(bandits):
        print("Mean estimatess for bandit {}: {}".format(i, bandit.p_estimate))

    print("Total reward earned:", rewards.sum())
    print("Overall win rate:", rewards.sum() / NUM_TRIALS)
    print("Number of times selecting each bandit:", [bandit.N for bandit in bandits])

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.figure()
    plt.plot(win_rates, label="win rates")
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), label="max probability")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    experiment()