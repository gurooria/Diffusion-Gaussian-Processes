import torch
import gpytorch
from exactGPs import ExactGP

## Evaluation metrics for all models
def evaluate_metrics(pred_mean, pred_var, test_y, mean_y, type='rmse'):
    def univariate_nll(test_y, pred_mean, pred_var):
        return 0.5 * torch.log(2 * torch.pi * pred_var) + (test_y - pred_mean)**2 / (2 * pred_var)

    if type == 'rmse':
        return torch.sqrt(torch.mean((pred_mean - mean_y)**2)) # RMSE is the error from the true mean only
    elif type == 'nll':
        nll = univariate_nll(test_y, pred_mean, pred_var) # uses test data points to compute NLL
        return nll.mean()
    

## Training function for exact GP models
def train_exact_gp(train_x, train_y, test_x, test_y, mean_y, lr:float, eval=True, num_trials:int=5, training_iter:int=200, report_iter:int=50, device:str=None):
    """
    Function to train an exact GP model.

    Args:
        train_x (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training output data.
        lr (float): Learning rate.
        eval (bool): Whether to evaluate the model.
        num_trials (int): Number of trials to run.
        training_iter (int): Number of training iterations.
        report_iter (int): Number of iterations to report progress.

    Returns:
        model (gpytorch.models.ExactGP): Trained GP model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): Trained likelihood.
        rmses (torch.Tensor): RMSEs of the trials.
        nlls (torch.Tensor): NLLs of the trials.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    rmses = torch.zeros(num_trials)
    nlls = torch.zeros(num_trials)
    best_model = None
    best_likelihood = None
    best_rmse = float('inf')
    best_nll = float('inf')

    if not eval:
        num_trials = 1

    # Repeat training for num_trials times
    for i in range(num_trials):
        print(f"-------- Trial {i+1} of {num_trials} --------")

        # Initialisations
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGP(train_x, train_y, likelihood).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)

        # Print trainable parameters
        print('Intial parameters of GP model - lengthscale: %.3f   noise: %.3f   outputscale: %.3f' % (
                    model.covar_module.base_kernel.lengthscale.item(), # kernel lengthscale
                    model.likelihood.noise.item(), # likelihood noise
                    model.covar_module.outputscale.item(), # kernel outputscale
                ))
        
        ## Training loop
        for j in range(training_iter):
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            output = model(train_x) # forward pass, output is a distribution over function values f(x) at train_x
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if j % report_iter == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f   outputscale: %.3f' % (
                    j + 1, training_iter, loss.item(),
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
        
        if eval:
            ## Evaluation GP
            print("Evaluating GP model...")
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                preds = likelihood(model(test_x))
                means = preds.mean
                alo_vars = likelihood(model(test_x)).variance - model(test_x).variance

            # RMSE
            rmse = evaluate_metrics(pred_mean=means, pred_var=alo_vars, test_y=test_y, mean_y=mean_y, type='rmse')
            rmses[i] = rmse

            # NLL
            nll = evaluate_metrics(pred_mean=means, pred_var=alo_vars, test_y=test_y, mean_y=mean_y, type='nll')
            nlls[i] = nll

            # Update best model
            if rmse < best_rmse:
                best_rmse = rmse
            if nll < best_nll:
                best_nll = nll
                best_model = model
                best_likelihood = likelihood
                print(f"Best model updated: RMSE: {rmse:.3f}   NLL: {nll:.3f}")
            
    if eval:
        print(f"RMSE: {rmses.mean()} ± {rmses.std()}")
        print(f"NLL: {nlls.mean()} ± {nlls.std()}")
        return best_model, best_likelihood, rmses, nlls
        
    else:
        return model, likelihood