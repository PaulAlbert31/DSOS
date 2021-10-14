import scipy.stats as stats
import torch
import matplotlib.pyplot as plt

def weighted_mean(x, w):
    return torch.sum(w * x.double()) / torch.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(torch.nn.Module):
    def __init__(self, max_iters=100,
                 n_components=2):
        super(BetaMixture1D, self).__init__()
        self.alphas = torch.arange(n_components, dtype=torch.float64) + 1
        self.betas = n_components - torch.arange(n_components, dtype=torch.float64) + 1
        self.weight = torch.ones(n_components, dtype=torch.float64) / n_components
        self.max_iters = max_iters
        self.eps_nan = 1e-12
        self.n_components = n_components

    def likelihood(self, x, y):
        return torch.tensor(stats.beta.pdf(x, self.alphas[y], self.betas[y]))#.cuda()

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)
    

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(self.n_components))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)
    
    def responsibilities(self, x):
        r = torch.tensor([], dtype=torch.float64)
        for i in range(self.n_components):
            r = torch.cat((r, self.weighted_likelihood(x, i)))
        r[r <= self.eps_nan] = self.eps_nan
        r = r.view(self.n_components, -1)
        r /= torch.sum(r, dim=0)
        return r

    def score_samples(self, x):
        return -torch.log(self.probability(x))

    def fit(self, x):
        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            for i in range(self.n_components):
                self.alphas[i], self.betas[i] = fit_beta_weighted(x, r[i])
            
            self.weight = torch.sum(r, dim=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def plot(self):
        x = torch.linspace(0, 1, 100)
        for i in range(self.n_components):
            plt.plot(x, self.weighted_likelihood(x, i), label='Low loss')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.legend()
        plt.title('BetaModel')
        plt.show()

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
