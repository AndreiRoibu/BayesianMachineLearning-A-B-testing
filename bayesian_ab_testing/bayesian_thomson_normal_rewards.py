import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import norm

np.random.seed(1)
NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]

class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean

        # prior for mu is N(0,1)
        self.m = 0 # the predicted mean of the mean of X
        self.lambda_ = 1 # assumed precision (same as variance=1)

        self.tau = 1
        self.N = 0

    def pull(self):
        # draws a sample from N(mean=true_mean, precision=tau)
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        # posterior
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        # the number of samples N is 1 in the bernoulli case 
        self.m = (self.tau * x + self.lambda_ * self.m) / ( self.tau + self.lambda_ )
        self.lambda_ += self.tau
        self.N += 1

def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    for bandit in bandits:
        y = norm.pdf(x, bandit.m, np.sqrt(1. / bandit.lambda_))
        plt.plot(x, y, label=f"Real mean: {bandit.true_mean:.2f}, Number of plays: {bandit.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()

def run_experiment():
    bandits = [Bandit(m) for m in BANDIT_MEANS]
    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    rewards = np.empty(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits])

        if i in sample_points:
            plot(bandits, i)

        x = bandits[j].pull()
        bandits[j].update(x)
        rewards[i] = x

    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    for m in BANDIT_MEANS:
        plt.plot(np.ones(NUM_TRIALS)*m)
    plt.show()

    return cumulative_average

if __name__ == '__main__':
    run_experiment()
