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
        self.p_estimate = 0.0
        self.N = 0.0 # number of samples so far
    
    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    times_explored = 0
    times_exploited = 0
    optimals = 0
    optimal_j = np.argmax([bandit.p for bandit in bandits])
    print("Original optimal j:", optimal_j)

    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select the next bandit
        # either randomly selecting one of the bandits
        # or selecting the best one based on p_estimate
        if np.random.random() < EPSILON:
            times_explored += 1
            j = np.random.randint(len(bandits))
        else:
            times_exploited += 1
            j = np.argmax([bandit.p_estimate for bandit in bandits])

        if j == optimal_j:
            optimals += 1
            
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
    print("Number of times explored:", times_explored)
    print("Number of times eploited:", times_exploited)
    print("Number of times selected optimal bandit:", optimals)

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