import torch
import gpytorch

def evaluate_metrics(pred_mean, pred_var, test_y, true_y, type='rmse'):
    def univariate_nll(true_y, pred_mean, pred_var):
        return 0.5 * torch.log(2 * torch.pi * pred_var) + (true_y - pred_mean)**2 / (2 * pred_var)

    if type == 'rmse':
        return torch.sqrt(torch.mean((pred_mean - true_y)**2)) # RMSE is the error from the true mean only
    elif type == 'nll':
        nll = univariate_nll(true_y, pred_mean, pred_var)
        return nll.mean()

def train_exact_gp(gp_model, train_x, train_y, lr, training_iter:200, report_iter=50):
    likelihood = gpytorch.likelihoods.GaussianLikelihood() # likelihood for mean GP
    model = gp_model(train_x, train_y, likelihood) # GP model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimises GP kernel parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # computes negative log marginal likelihood

    # Print trainable parameters
    print('Intial parameters of GP model - lengthscale: %.3f   noise: %.3f   outputscale: %.3f' % (
                model.covar_module.base_kernel.lengthscale.item(), # kernel lengthscale
                model.likelihood.noise.item(), # likelihood noise
                model.covar_module.outputscale.item(), # kernel outputscale
            ))

    # Train model via MLL optimisation
    model.train()
    likelihood.train()
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x) # forward pass, output is a distribution over function values f(x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if i % report_iter == 0:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f   outputscale: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(), # kernel lengthscale
                model.likelihood.noise.item(), # likelihood noise
                model.covar_module.outputscale.item(), # kernel outputscale
            ))

    # Print trained parameters
    print('Trained parameters of GP model - lengthscale: %.3f   noise: %.3f   outputscale: %.3f' % (
                model.covar_module.base_kernel.lengthscale.item(), # kernel lengthscale
                model.likelihood.noise.item(), # likelihood noise
                model.covar_module.outputscale.item(), # kernel outputscale
            ))

    return model, likelihood

def eval_exact_gp(model, likelihood, test_x, test_y, true_y):
    # Evaluate model
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(test_x))
        means = preds.mean
        total_vars = preds.variance

    # RMSE
    rmse = evaluate_metrics(pred_mean=means, pred_var=total_vars, test_y=test_y, true_y=true_y, type='rmse')
    print(f"RMSE: {rmse}")

    # NLL
    nll = evaluate_metrics(pred_mean=means, pred_var=total_vars, test_y=test_y, true_y=true_y, type='nll')
    print(f"NLL: {nll}")

    return rmse, nll