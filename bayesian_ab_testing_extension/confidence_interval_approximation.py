import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import beta, norm

ITERATION = 501
TRUE_CTR = 0.5
a,b = 1, 1 # beta priors
plot_indices = (10, 20, 30, 50, 100, 200, 500)
data = np.empty(ITERATION)
for i in range(ITERATION):
    x = 1 if np.random.random() < TRUE_CTR else 0
    data[i] = x
    a += x
    b += 1-x

    if i in plot_indices:
        # maximum likelihood of CTR
        p = data[:i].mean()
        n = i+1 # number of samples collected so far
        std = np.sqrt( p * (1-p) / n)

        # gaussian
        x_gaussian = np.linspace(0,1,200)
        g = norm.pdf(x_gaussian, loc=p, scale=std)
        plt.plot(x_gaussian, g, label="Gaussian Approximation")

        # beta
        posterior = beta.pdf(x_gaussian, a=a, b=b)
        plt.plot(x_gaussian, posterior, label='Beta Posterior')
        plt.legend()
        plt.title("N = %s" % n)
        plt.show()