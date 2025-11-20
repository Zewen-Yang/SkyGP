import numpy as np
from scipy.linalg import solve_triangular


def sigmoid(x, temp=10.0):
    """Sigmoid with temperature scaling (controls growth speed)."""
    return 1 / (1 + np.exp(-x / temp))


class LocalGP_MOE:
    """
    Local GP Mixture-of-Experts.
    - Experts are selected by kernel similarity to their running centers.
    - When full, an expert may replace a farthest sample or spawn a new expert
      depending on a kernel similarity threshold to the center.
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        max_data_per_expert=5000,
        nearest_k=2,
        max_experts=4,
        replacement=False,
        pretrained_params=None,
        threshold=1.0,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_data = max_data_per_expert
        self.max_experts = max_experts
        self.nearest_k = nearest_k

        # Per-expert storage
        self.X_list = []
        self.Y_list = []
        self.localCount = []
        self.expert_centers = []
        self.drop_centers = []
        self.drop_counts = []
        self.model_params = {}
        self.L_all = []
        self.alpha_all = []

        self.pretrained_params = pretrained_params
        self.kernel_threshold = threshold

        # Runtime state
        self.last_sorted_experts = None
        self.last_prediction_cache = {}   # Cache: k_star, v, mu_part per expert
        self.expert_usage_counts = []     # Usage count per expert
        self.replacement = replacement    # Enable replacement policy

        self.expert_usage_stats = []      # {'used': ..., 'considered': ...}
        self.expert_weights = []          # Weight per expert (if used externally)

        self.last_x = None                # Last input, for distance-based logic if needed
        self.last_expert_idx = None       # Index of last-used expert

        self.expert_id_counter = 0

    # ------------------------------------------------------------------ #
    # Kernel & parameter init
    # ------------------------------------------------------------------ #

    def kernel_np(self, X1, X2, lengthscale, sigma_f):
        """Squared exponential (RBF) kernel."""
        X1_scaled = X1 / lengthscale[:, None]
        X2_scaled = X2 / lengthscale[:, None]
        dists = np.sum((X1_scaled[:, :, None] - X2_scaled[:, None, :]) ** 2, axis=0)
        return sigma_f**2 * np.exp(-0.5 * dists)

    def init_model_params(self, model_id, pretrained_params=None):
        """Initialize hyperparameters (prefer provided/pretrained params)."""
        if pretrained_params:
            print(f"Using pretrained parameters for model {model_id}.")
            self.pretrained_params = pretrained_params
            outputscale, noise, lengthscale = pretrained_params
            log_sigma_f = np.log(outputscale.flatten())
            log_sigma_n = np.log(noise.flatten())
            if lengthscale.ndim == 2 and lengthscale.shape[1] == self.y_dim:
                log_lengthscale = np.log(lengthscale)
            else:
                log_lengthscale = np.log(lengthscale.squeeze())
        elif self.pretrained_params:
            outputscale, noise, lengthscale = self.pretrained_params
            log_sigma_f = np.log(outputscale.flatten())
            log_sigma_n = np.log(noise.flatten())
            if lengthscale.ndim == 2 and lengthscale.shape[1] == self.y_dim:
                log_lengthscale = np.log(lengthscale)
            else:
                log_lengthscale = np.log(lengthscale.squeeze())
        else:
            log_sigma_f = np.log(np.ones(self.y_dim))
            log_sigma_n = np.log(np.ones(self.y_dim) * 0.01)
            log_lengthscale = np.log(
                np.ones((self.x_dim,)) if self.y_dim == 1
                else np.ones((self.x_dim, self.y_dim))
            )

        self.model_params[model_id] = {
            "log_sigma_f": log_sigma_f,
            "log_sigma_n": log_sigma_n,
            "log_lengthscale": log_lengthscale,
        }

    # ------------------------------------------------------------------ #
    # Expert lifecycle
    # ------------------------------------------------------------------ #

    def _create_new_expert(self):
        """Create and register a new expert; return its index."""
        self.X_list.append(np.zeros((self.x_dim, self.max_data)))
        self.Y_list.append(np.zeros((self.y_dim, self.max_data)))
        self.localCount.append(0)
        self.expert_centers.append(np.zeros(self.x_dim))
        self.drop_centers.append(np.zeros(self.x_dim))
        self.drop_counts.append(0)
        self.L_all.append(np.zeros((self.max_data, self.max_data)))
        self.alpha_all.append(np.zeros((self.max_data, self.y_dim)))
        self.expert_usage_counts.append(0)
        self.expert_usage_stats.append({"used": 0, "considered": 0})
        self.expert_weights.append(1.0)

        self.init_model_params(len(self.X_list) - 1)
        return len(self.X_list) - 1

    def _replace_expert(self, idx):
        """Hard-reset an expert slot."""
        self.X_list[idx] = np.zeros((self.x_dim, self.max_data))
        self.Y_list[idx] = np.zeros((self.y_dim, self.max_data))
        self.localCount[idx] = 0
        self.expert_centers[idx] = np.zeros(self.x_dim)
        self.drop_centers[idx] = np.zeros(self.x_dim)
        self.drop_counts[idx] = 0
        self.L_all[idx] = np.zeros((self.max_data, self.max_data))
        self.alpha_all[idx] = np.zeros((self.max_data, self.y_dim))
        self.expert_usage_counts[idx] = 0
        self.init_model_params(idx)

    # ------------------------------------------------------------------ #
    # Data ingestion
    # ------------------------------------------------------------------ #

    def add_point(self, x, y):
        """
        Add (x, y) to the current/last expert if possible.
        If expert is full:
          - If kernel similarity to center > threshold: replace farthest sample.
          - Else: create a new expert and insert there.
        """
        x = np.asarray(x)
        y = np.asarray(y).reshape(-1)

        x_uncat = x.copy()
        y_uncat = y.copy()

        # If we don't have a cached order, recompute by kernel similarity to centers
        expert_order = self.last_sorted_experts if self.last_sorted_experts is not None else []
        if self.last_sorted_experts is None:
            print("🔄 Recalculating expert order...")
            outputscale, noise, lengthscale = self.pretrained_params
            sigma_f = np.atleast_1d(outputscale)[0]
            lengthscale = lengthscale if lengthscale.ndim == 1 else lengthscale[:, 0]
            if len(self.expert_centers) == 0:
                expert_order = []
            else:
                dists = [
                    (self.kernel_np(x_uncat[None, :], c[None, :], lengthscale, sigma_f)[0, 0], i)
                    for i, c in enumerate(self.expert_centers)
                ]
                dists.sort()
                expert_order = [i for _, i in dists]

        # Choose the last-used expert as default; fallback to 0/new
        model = self.last_expert_idx
        if model is None:
            model = 0

        if model >= len(self.X_list):
            model = self._create_new_expert()

        idx = self.localCount[model]

        # Case 1: capacity available → append
        if idx < self.max_data:
            self.X_list[model][:, idx] = x_uncat
            self.Y_list[model][:, idx] = y_uncat
            self.localCount[model] += 1

            # Update running center
            if idx == 0:
                self.expert_centers[model] = x_uncat
            else:
                self.expert_centers[model] = (self.expert_centers[model] * idx + x_uncat) / (idx + 1)

            self.update_param_incremental(x_uncat, y_uncat, model)
            return

        # Case 2: expert is full → decide replace vs. new expert based on kernel similarity
        outputscale, noise, lengthscale = self.pretrained_params
        sigma_f = np.atleast_1d(outputscale)[0]
        lengthscale = lengthscale if lengthscale.ndim == 1 else lengthscale[:, 0]

        kernel_sim = self.kernel_np(
            x_uncat[None, :], self.expert_centers[model][None, :], lengthscale, sigma_f
        )[0, 0]

        if kernel_sim > self.kernel_threshold:
            # Replace the farthest-from-center sample
            stored = self.X_list[model][:, :self.max_data]
            dists = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
            max_idx = np.argmax(dists)

            x_old = self.X_list[model][:, max_idx].copy()
            y_old = self.Y_list[model][:, max_idx].copy()

            self.X_list[model][:, max_idx] = x_uncat
            self.Y_list[model][:, max_idx] = y_uncat

            # Update center incrementally (approx.)
            self.expert_centers[model] += (x_uncat - x_old) / self.max_data
            self.drop_centers[model] = x_old
            self.drop_counts[model] += 1

            self.update_param(model)
            return

        # Otherwise spawn a new expert and insert there
        new_model = self._create_new_expert()
        self._insert_into_expert(new_model, x_uncat, y_uncat)
        return

    def _insert_new_expert_near(self, near_idx):
        """Placeholder: always create a new expert (no positional logic here)."""
        return self._create_new_expert()

    def _insert_into_expert(self, model, x, y):
        """Insert (x, y) into the given expert and update parameters incrementally."""
        idx = self.localCount[model]
        self.X_list[model][:, idx] = x
        self.Y_list[model][:, idx] = y
        self.localCount[model] += 1
        self.expert_centers[model] = x if idx == 0 else (self.expert_centers[model] * idx + x) / (idx + 1)
        self.update_param_incremental(x, y, model)

    # ------------------------------------------------------------------ #
    # GP parameter updates
    # ------------------------------------------------------------------ #

    def update_param(self, model):
        """Full recomputation of Cholesky factor L and alpha for expert `model`."""
        p = 0
        idx = self.localCount[model]
        params = self.model_params[model]
        sigma_f = np.exp(params["log_sigma_f"][p])
        sigma_n = np.exp(params["log_sigma_n"][p])
        lengthscale = (
            np.exp(params["log_lengthscale"]) if self.y_dim == 1
            else np.exp(params["log_lengthscale"][:, p])
        )

        X_subset = self.X_list[model][:, :idx]
        Y_subset = self.Y_list[model][p, :idx]

        K = self.kernel_np(X_subset, X_subset, lengthscale, sigma_f)
        K[np.diag_indices_from(K)] += sigma_n**2

        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(idx))
        except np.linalg.LinAlgError:
            print(f"⚠️ Cholesky failed for model {model}, using identity fallback")
            L = np.eye(idx)

        self.L_all[model][:idx, :idx] = L
        aux_alpha = solve_triangular(L, Y_subset, lower=True)
        self.alpha_all[model][:idx, p] = solve_triangular(L.T, aux_alpha, lower=False)

    def update_param_incremental(self, x, y, model):
        """Rank-1 incremental update of L and alpha when appending one sample."""
        p = 0
        idx = self.localCount[model]
        if idx == 0:
            return  # Nothing to update

        params = self.model_params[model]
        sigma_f = np.exp(params["log_sigma_f"][p])
        sigma_n = np.exp(params["log_sigma_n"][p])
        lengthscale = (
            np.exp(params["log_lengthscale"]) if self.y_dim == 1
            else np.exp(params["log_lengthscale"][:, p])
        )

        if idx == 1:
            kxx = self.kernel_np(x[:, None], x[:, None], lengthscale, sigma_f)[0, 0] + sigma_n**2
            L = np.sqrt(kxx)
            self.L_all[model][0, 0] = L
            self.alpha_all[model][0, p] = y / (L * L)
            return

        X_prev = self.X_list[model][:, : idx - 1]
        y_vals = self.Y_list[model][p, :idx]

        # Use cached k_star and v if available from the last prediction
        cached = self.last_prediction_cache.get(model, {}).get(p, None)
        if cached is not None and cached["k_star"].shape[0] == idx - 1:
            b = cached["k_star"]
            Lx = cached["v"]
        else:
            b = self.kernel_np(X_prev, x[:, None], lengthscale, sigma_f).flatten()
            L_prev = self.L_all[model][: idx - 1, : idx - 1]
            Lx = solve_triangular(L_prev, b, lower=True)

        c = self.kernel_np(x[:, None], x[:, None], lengthscale, sigma_f)[0, 0] + sigma_n**2
        Lii = np.sqrt(max(c - np.dot(Lx, Lx), 1e-9))

        self.L_all[model][: idx - 1, idx - 1] = 0.0
        self.L_all[model][idx - 1, : idx - 1] = Lx
        self.L_all[model][idx - 1, idx - 1] = Lii

        L_now = self.L_all[model][:idx, :idx]
        aux_alpha = solve_triangular(L_now, y_vals, lower=True)
        self.alpha_all[model][:idx, p] = solve_triangular(L_now.T, aux_alpha, lower=False)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, x_query):
        """
        Predict using top-k experts ranked by kernel similarity to x_query.
        Fuse expert predictions with similarity-weighted averaging.
        """
        self.last_prediction_cache.clear()

        if len(self.expert_centers) == 0:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        outputscale, noise, lengthscale = self.pretrained_params
        sigma_f = np.atleast_1d(outputscale)[0]
        lengthscale = lengthscale if lengthscale.ndim == 1 else lengthscale[:, 0]

        # (1) Compute kernel similarity to all expert centers
        dists = [
            (
                self.kernel_np(center[:, None], x_query[:, None], lengthscale, sigma_f)[0, 0],
                idx,
            )
            for idx, center in enumerate(self.expert_centers)
        ]
        if not dists:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        # (2) Select top-k by similarity (descending)
        dists.sort(reverse=True)
        selected = [idx for _, idx in dists[: self.nearest_k]]
        self.last_sorted_experts = selected
        self.last_expert_idx = selected[0]
        self.last_x = x_query

        # (3) Per-expert GP predictions
        mus, vars_ = [], []
        for idx in selected:
            L = self.L_all[idx]
            alpha = self.alpha_all[idx]
            X_snapshot = self.X_list[idx][:, : self.localCount[idx]]
            n_valid = self.localCount[idx]

            mu = np.zeros(self.y_dim)
            var = np.zeros(self.y_dim)

            for p in range(self.y_dim):
                params = self.model_params[idx]
                sigma_f = np.exp(params["log_sigma_f"][p])
                sigma_n = np.exp(params["log_sigma_n"][p])
                lengthscale = (
                    np.exp(params["log_lengthscale"][:, p]) if self.y_dim > 1
                    else np.exp(params["log_lengthscale"])
                )

                k_star = self.kernel_np(X_snapshot, x_query[:, None], lengthscale, sigma_f).flatten()
                k_xx = sigma_f**2

                mu[p] = np.dot(k_star, alpha[:n_valid, p])
                v = solve_triangular(L[:n_valid, :n_valid], k_star, lower=True)
                var[p] = k_xx - np.sum(v**2)

                # Cache for potential incremental update
                if idx not in self.last_prediction_cache:
                    self.last_prediction_cache[idx] = {}
                self.last_prediction_cache[idx][p] = {
                    "k_star": k_star.copy(),
                    "v": v.copy(),
                    "mu_part": mu[p],
                }

            mus.append(mu)
            vars_.append(var)

        # (4) Mixture fusion with similarity weights
        mus = np.stack(mus)       # (k, y_dim)
        vars_ = np.stack(vars_)   # (k, y_dim)

        # Similarity scores correspond to the already-sorted top-k list
        similarities = np.array([sim for sim, _ in dists[: self.nearest_k]])
        if np.sum(similarities) == 0:
            similarities = np.ones_like(similarities)

        weights = similarities / (np.sum(similarities) + 1e-8)
        weights = weights[:, None]  # (k, 1)

        mu_moe = np.sum(weights * mus, axis=0)
        var_moe = np.sum(weights * vars_, axis=0)  # weighted expected variance

        return mu_moe, var_moe