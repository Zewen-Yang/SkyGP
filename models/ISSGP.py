import numpy as np
from numpy.linalg import cholesky, LinAlgError, solve
import pdb

class ISSGP:
    def __init__(self, input_dim, output_dim, num_features, hyper):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d = num_features

        self.sigmaL = hyper[:-2]
        self.sigmaF = hyper[-2]
        self.sigmaN = hyper[-1]

        self.R = self.sigmaN * np.eye(2 * self.d)
        self.b = np.zeros((2 * self.d, 1))
        self.w = np.zeros((2 * self.d, 1))

        sigma = np.diag(1.0 / self.sigmaL**2)
        mu = np.zeros((self.d, self.input_dim))
        self.omega = np.empty((self.d, self.input_dim))

        for i in range(self.d):
            self.omega[i, :] = np.random.multivariate_normal(mean=mu[i], cov=sigma)
        # from scipy.io import loadmat

        # # 读取 .mat 文件
        # mat = loadmat('omega_fixed.mat')
        # omega = mat['omega']  # numpy array of shape (d, D)

        # # 使用在 ISSGP 模型中
        # self.omega = omega  # 覆盖随机生成的 omega

        # #Omega stats
        # print("[INIT] omega shape:", self.omega.shape)
        # print("[INIT] omega mean: {:.4f}, std: {:.4f}, norm: {:.4f}".format(
        #     np.mean(self.omega), np.std(self.omega), np.linalg.norm(self.omega)))

        self.A = self.R.T @ self.R

    def _phi(self, x):
        x = x.flatten()
        # pdb.set_trace()
        proj = self.omega @ x
        cos_part = np.cos(proj)
        sin_part = np.sin(proj)
        phi_raw = np.vstack((cos_part.reshape(-1,1), sin_part.reshape(-1,1)))
        phi = phi_raw * self.sigmaF / np.sqrt(self.d)
        return phi

    def update(self, x, y):
        phi = self._phi(x)
        self.A += phi @ phi.T
        try:
            self.R = cholesky(self.A)
        except LinAlgError:
            print("[UPDATE] Cholesky failed, regularizing A...")
            self.A += 1e-6 * np.eye(self.A.shape[0])
            self.R = cholesky(self.A)

        self.b += phi * y
        try:
            temp = solve(self.R, self.b)
            self.w = solve(self.R.T, temp)
        except LinAlgError:
            self.R += 1e-6 * np.eye(self.R.shape[0])
            self.w = solve(self.R.T, solve(self.R, self.b))

    def predict(self, x):
        phi = self._phi(x)
        mu = float(phi.T @ self.w)
        v = solve(self.R, phi)
        var = self.sigmaN ** 2 * float(v.T @ v)
        var = np.clip(var, 1e-6, None)
        # Debug prediction

        return mu, var
