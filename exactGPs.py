import gpytorch

## Exact GP models for GP baselines, computes analytically exact posterior and marginal likelihood
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood) # pass train_x, train_y, likelihood to parent class
        # define components of prior GP, p(f(x))
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        # returns distribution over function values f(x), prior distirbution
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)