from bayesian_ab_testing.ucb1 import NUM_TRIALS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import beta

np.random.seed(1)
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0

    def pull(self):
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1

def plot(bandits, trial):
    x = np.linspace(0,1,200)
    for bandit in bandits:
        y = beta.pdf(x, bandit.a, bandit.b)
        plt.plot(x, y, label=f"real p: {bandit.p:.4f}, win rate = {bandit.a - 1}/{bandit.N}")
    plt.title(f"Bandit distribution after {trial} trials")
    plt.legend()
    plt.show()

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        j = np.argmax([bandit.sample() for bandit in bandits]) # Thomson Sampling
        if i in sample_points:
            plot(bandits, trial=i)

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        # update rewards
        rewards[i] = x
        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [bandit.N for bandit in bandits])


if __name__ == "__main__":
    experiment()
