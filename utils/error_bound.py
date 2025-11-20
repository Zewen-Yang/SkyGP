import numpy as np

class ErrorBoundCalculator:
    def __init__(self, input_dim, domain_size=5.0, tau=0.05, delta=0.01,
                 L_total=2.0, omega_sigma=0.1):
        self.d = input_dim
        self.r = domain_size
        self.tau = tau
        self.delta = delta
        self.L_total = L_total
        self.omega_sigma = omega_sigma
        self.beta = self.compute_beta()

    def compute_beta(self):
        M_tau = (1 + self.r / self.tau) ** self.d
        return 2 * np.log(M_tau / self.delta)

    def gamma(self):
        return self.L_total * self.tau + np.sqrt(self.beta) * self.omega_sigma

    def compute_bound(self, std):
        """给定某点的 GP 标准差，返回误差上界"""
        return np.sqrt(self.beta) * std + self.gamma()