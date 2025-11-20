# gp_trainer.py
import torch
import gpytorch

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GP_Approximator:
    def __init__(self, train_x, train_y, lr=0.005, num_iters=200):
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPRegressionModel(train_x, train_y, self.likelihood)
        self.lr = lr
        self.num_iters = num_iters

    def train(self):
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.num_iters):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0 or i == 0:
                print(f"Iter {i+1}/{self.num_iters} - Loss: {loss.item():.3f}")

        self.print_parameters()
        return (
            self.model.covar_module.outputscale.detach().float(),
            self.likelihood.noise.detach().float(),
            self.model.covar_module.base_kernel.lengthscale.detach().float().squeeze()
        )

    def print_parameters(self):
        print("\n📊 Learned GP Parameters:")
        print("Lengthscale:", self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
        print("Outputscale:", self.model.covar_module.outputscale.item())
        print("Noise:", self.likelihood.noise.item())