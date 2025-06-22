import gpytorch
import torch

class ApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing_points, input_dim, mean_type = 'linear', matern_nu=[0.5, 0.5, 0.5], seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        inducing_points = torch.rand(num_inducing_points, input_dim)
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.input_dim = input_dim # dimension of the input (concatenation of x, t, y_t)
        # Initialise mean module
        if mean_type == 'linear':
            self.mean_module = gpytorch.means.LinearMean(input_size=self.input_dim)
        elif mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(input_size=self.input_dim)
        elif mean_type == 'zero':
            self.mean_module = gpytorch.means.ZeroMean(input_size=self.input_dim)
        else:
            raise ValueError(f"Mean type {mean_type} not supported")
        # Initialise covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ProductKernel(
                gpytorch.kernels.MaternKernel(nu=matern_nu[0], active_dims=list(range(input_dim-2))),  # x
                gpytorch.kernels.MaternKernel(nu=matern_nu[1], active_dims=input_dim-2),  # t
                gpytorch.kernels.MaternKernel(nu=matern_nu[2], active_dims=input_dim-1),   # y_t
            ), 
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)