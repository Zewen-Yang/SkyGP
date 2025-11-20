import torch
import torch.nn as nn
from torch.linalg import cholesky, solve


class StreamingSGP(nn.Module):
    def __init__(self, X, Y, Z, alpha=0.5, M=20 ,mean_function=None, device='cpu',pretrained_params=None):
        
        super().__init__()
        self.device = device
        self.alpha = alpha

        self.X = X.to(device)
        self.Y = Y.to(device)
        self.Z = Z.to(device)

        self.N, self.D = self.X.shape
        self.M = M
        self.Z = nn.Parameter(Z.to(device))

        # log-parameters (lengthscale, variance, noise)
        if pretrained_params is not None:
            outputscale, noise, lengthscale = pretrained_params

            lengthscale = torch.tensor(lengthscale, dtype=torch.float32, device=device)
            outputscale = torch.tensor(outputscale, dtype=torch.float32, device=device)
            noise = torch.tensor(noise, dtype=torch.float32, device=device)

            self.log_lengthscale = nn.Parameter(lengthscale.log(), requires_grad=False)
            self.log_variance = nn.Parameter(outputscale.log(), requires_grad=False)
            self.log_noise = nn.Parameter(noise.log(), requires_grad=False)
        else:
            self.log_lengthscale = nn.Parameter(torch.zeros(self.D, device=device))
            self.log_variance = nn.Parameter(torch.tensor(0.0, device=device))
            self.log_noise = nn.Parameter(torch.tensor(-4.6, device=device))
        
        self.y_mean = self.Y.mean(0, keepdim=True)
        self.Y_init_centered = self.Y - self.y_mean

        self.mean_function = lambda X: self.y_mean.expand(X.shape[0], -1)
        
        # 初始化变分分布参数（可学习）
        self.q_mu = nn.Parameter(torch.zeros(self.M, 1, device=self.device))  # 均值
        self.q_S = nn.Parameter(torch.eye(self.M, device=self.device))        # 协方差矩阵

        # 保存上一批的后验作为当前先验
        self.q_mu_old = torch.zeros_like(self.q_mu)
        self.q_S_old = torch.eye(self.M, device=self.device)
        
        self.Z_old = self.Z.detach().clone()
        self.Kaa_old = self.kernel(self.Z_old, self.Z_old) + 1e-4 * torch.eye(self.M, device=self.device)
        self.window_size = 1000  # 默认值，可修改

    def kernel(self, X1, X2):
        lengthscale = self.log_lengthscale.exp()
        variance = self.log_variance.exp()
        X1_scaled = X1 / lengthscale
        X2_scaled = X2 / lengthscale
        sqdist = (X1_scaled ** 2).sum(-1)[:, None] + (X2_scaled ** 2).sum(-1)[None, :] - 2 * X1_scaled @ X2_scaled.T
        return variance * torch.exp(-0.5 * sqdist)

    # def _common_terms(self):
    #     sigma2 = self.log_noise.exp() ** 2
    #     alpha = self.alpha
        
    #     Kf_diag = self.kernel(self.X, self.X).diagonal()
    #     Kbf = self.kernel(self.Z, self.X)
    #     Kbb = self.kernel(self.Z, self.Z) + 1e-4 * torch.eye(self.M, device=self.device)

    #     err = self.Y - self.mean_function(self.X)
    #     # print("err:", err)
    #     Lb = cholesky(Kbb)
    #     Lb_inv_Kbf = solve(Lb, Kbf)

    #     Qff_diag = torch.sum(Lb_inv_Kbf ** 2, dim=0)
    #     Dff = sigma2 + self.alpha * (Kf_diag - Qff_diag)

    #     Lb_inv_Kbf_LDff = Lb_inv_Kbf / Dff.sqrt().unsqueeze(0)
    #     BBT = Lb_inv_Kbf_LDff @ Lb_inv_Kbf_LDff.T + torch.eye(self.M, device=self.device)
    #     LD = cholesky(BBT + 1e-6 * torch.eye(self.M, device=self.device))

    #     Sinv_y = self.Y / Dff.view(-1, 1)
    #     c = Lb_inv_Kbf @ Sinv_y
    #     LDinv_c = solve(LD, c)

    #     return Kbf, Kbb, Lb, BBT, LD, LDinv_c, err, Dff, sigma2
    
    def _common_terms(self, X, Y):
        sigma2 = self.log_noise.exp() ** 2
        alpha = self.alpha
        
        Kf_diag = self.kernel(X, X).diagonal()
        Kbf = self.kernel(self.Z, X)
        Kbb = self.kernel(self.Z, self.Z) + 1e-4 * torch.eye(self.M, device=self.device)

        err = Y - self.mean_function(X)
        Lb = cholesky(Kbb)
        Lb_inv_Kbf = solve(Lb, Kbf)

        Qff_diag = torch.sum(Lb_inv_Kbf ** 2, dim=0)
        Dff = sigma2 + alpha * (Kf_diag - Qff_diag)

        Lb_inv_Kbf_LDff = Lb_inv_Kbf / Dff.sqrt().unsqueeze(0)
        BBT = Lb_inv_Kbf_LDff @ Lb_inv_Kbf_LDff.T + torch.eye(self.M, device=self.device)
        LD = cholesky(BBT + 1e-6 * torch.eye(self.M, device=self.device))

        # 添加旧诱导相关核矩阵
        Kba = self.kernel(self.Z, self.Z_old)  # shape [M, M]
        Sainv_ma = torch.linalg.solve(self.q_S_old, self.q_mu_old)  # S⁻¹ * m
        c2 = Kba @ Sainv_ma  # [M,1]

        c1 = Lb_inv_Kbf @ (Y / Dff.view(-1, 1))
        c = c1 + c2  # 新的右侧项
        LDinv_c = solve(LD, c)

        return Kbf, Kbb, Lb, BBT, LD, LDinv_c, err, Dff, sigma2


    # def maximum_log_likelihood_objective(self, X_batch, Y_batch):
    #     Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff, sigma2 = self._common_terms()
    #     N = self.N
    #     alpha = self.alpha

    #     # ELBO 的对数似然项（按原来的）
    #     term1 = -0.5 * N * torch.log(torch.tensor(2 * torch.pi, device=self.device))
    #     term2 = -0.5 * torch.sum(err ** 2 / Dff.view(-1, 1))
    #     term3 = 0.5 * torch.sum(LDinv_c ** 2)
    #     term4 = -0.5 * torch.sum(torch.log(Dff))
    #     term5 = -torch.sum(torch.log(torch.diagonal(LD)))
    #     term6 = -0.5 * (1 - alpha) / alpha * torch.sum(torch.log(Dff / sigma2))

    #     log_likelihood = term1 + term2 + term3 + term4 + term5 + term6

    #     # KL[q(u) || q_old(u)] —— 关键
    #     kl_term = self.kl_divergence_q_qold()

    #     return log_likelihood - kl_term  # maximize logL - KL
    
    def maximum_log_likelihood_objective(self, X, Y):
        Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff, sigma2 = self._common_terms(X, Y)
        N = X.shape[0]
        alpha = self.alpha

        term1 = -0.5 * N * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        term2 = -0.5 * torch.sum(err ** 2 / Dff.view(-1, 1))
        term3 = 0.5 * torch.sum(LDinv_c ** 2)
        term4 = -0.5 * torch.sum(torch.log(Dff))
        term5 = -torch.sum(torch.log(torch.diagonal(LD)))
        term6 = -0.5 * (1 - alpha) / alpha * torch.sum(torch.log(Dff / sigma2))

        log_likelihood = term1 + term2 + term3 + term4 + term5 + term6
        kl_term = self.kl_divergence_q_qold()

        return log_likelihood - kl_term

    def forward(self, X_batch=None, Y_batch=None):
        if X_batch is None or Y_batch is None:
            X_batch = self.X
            Y_batch = self.Y
        return -self.maximum_log_likelihood_objective(X_batch, Y_batch)

    def predict_f(self, Xnew, full_cov=False):
        jitter = 1e-4

        Kbs = self.kernel(self.Z, Xnew)

        # ✅ 传入当前训练数据 self.X 和 self.Y
        Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff, sigma2 = self._common_terms(self.X, self.Y)

        Lb_inv_Kbs = solve(Lb, Kbs)
        LDinv_Lbinv_Kbs = solve(LD, Lb_inv_Kbs)
        mean = (LDinv_Lbinv_Kbs.T @ LDinv_c).T

        if full_cov:
            Kss = self.kernel(Xnew, Xnew) + jitter * torch.eye(Xnew.shape[0], device=self.device)
            var1 = Kss
            var2 = -Lb_inv_Kbs.T @ Lb_inv_Kbs
            var3 = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, Xnew).diagonal() + jitter
            var2 = -torch.sum(Lb_inv_Kbs ** 2, dim=0)
            var3 = torch.sum(LDinv_Lbinv_Kbs ** 2, dim=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var

    def update(self, x_new, y_new):
        self.X = torch.cat([self.X, x_new], dim=0)[-self.window_size:]
        self.Y = torch.cat([self.Y, y_new], dim=0)[-self.window_size:]
        self.N = self.X.shape[0]
        
    def update_q_old(self):
        self.q_mu_old = self.q_mu.detach().clone()
        self.q_S_old = self.q_S.detach().clone()
        self.Z_old = self.Z.detach().clone()
        self.Kaa_old = self.kernel(self.Z_old, self.Z_old).detach() + 1e-4 * torch.eye(self.M, device=self.device)

        
    def kl_divergence_q_qold(self):
        q_mu = self.q_mu
        q_S = self.q_S
        p_mu = self.q_mu_old
        p_S = self.q_S_old

        M = self.M

        # 为防止奇异，加 jitter
        jitter = 1e-6 * torch.eye(M, device=self.device)
        q_S_stable = q_S + jitter
        p_S_stable = p_S + jitter

        q_L = cholesky(q_S_stable)
        p_S_inv = torch.inverse(p_S_stable)

        trace_term = torch.trace(p_S_inv @ q_S_stable)
        mean_diff = q_mu - p_mu
        mean_term = mean_diff.T @ p_S_inv @ mean_diff
        log_det_ratio = torch.logdet(p_S_stable) - torch.logdet(q_S_stable)

        kl = 0.5 * (trace_term + mean_term - M + log_det_ratio)
        return kl.squeeze()
