from pyexpat import model
import numpy as np
import random
from scipy.linalg import solve_triangular


def sigmoid(x, temp=10.0):
    """Sigmoid function with temperature scaling."""
    return 1 / (1 + np.exp(-x / temp))


class SkyGP_MOE:
    """
    SkyGP_MOE: A Mixture-of-Experts Gaussian Process model.
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
        timescale=0.03,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_data = max_data_per_expert
        self.max_experts = max_experts
        self.nearest_k = nearest_k

        # Expert storage
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

        # Caches and tracking
        self.last_sorted_experts = None
        self.last_prediction_cache = {}
        self.expert_usage_counts = []

        # Replacement and maintenance flags
        self.replacement = replacement

        # Expert metadata
        self.expert_creation_order = []
        self.expert_usage_stats = []
        self.expert_weights = []

        self.last_x = None
        self.last_expert_idx = None

        self.expert_dict = {}
        self.expert_id_counter = 0
        self.global_time_index = 0
        self.expert_predict_count = []
        self.expert_last_used_time = []

        self.timescale = timescale

    # ----------------------------------------------------------------------
    # Kernel and parameter initialization
    # ----------------------------------------------------------------------

    def kernel_np(self, X1, X2, lengthscale, sigma_f):
        """Squared exponential (RBF) kernel."""
        X1_scaled = X1 / lengthscale[:, None]
        X2_scaled = X2 / lengthscale[:, None]
        dists = np.sum((X1_scaled[:, :, None] - X2_scaled[:, None, :]) ** 2, axis=0)
        return sigma_f**2 * np.exp(-0.5 * dists)

    def init_model_params(self, model_id, pretrained_params=None):
        """Initialize model hyperparameters for each expert."""
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
                np.ones((self.x_dim,)) if self.y_dim == 1 else np.ones((self.x_dim, self.y_dim))
            )

        self.model_params[model_id] = {
            "log_sigma_f": log_sigma_f,
            "log_sigma_n": log_sigma_n,
            "log_lengthscale": log_lengthscale,
        }

    # ----------------------------------------------------------------------
    # Expert creation and management
    # ----------------------------------------------------------------------

    def _create_new_expert(self, model_id=None):
        """Create and initialize a new expert."""
        if model_id is None:
            model_id = self.expert_id_counter
            self.expert_id_counter += 1

        self.X_list.append(np.zeros((self.x_dim, self.max_data)))
        self.Y_list.append(np.zeros((self.y_dim, self.max_data)))
        self.localCount.append(0)
        self.expert_centers.append(np.zeros(self.x_dim))
        self.drop_centers.append(np.zeros(self.x_dim))
        self.drop_counts.append(0)
        self.L_all.append(np.zeros((self.max_data, self.max_data)))
        self.alpha_all.append(np.zeros((self.max_data, self.y_dim)))
        self.expert_usage_counts.append(0)
        self.expert_creation_order.append(model_id)
        self.expert_usage_stats.append({"used": 0, "considered": 0})
        self.expert_weights.append(1.0)
        self.init_model_params(model_id)
        self.expert_predict_count.append(0)
        self.expert_last_used_time.append(self.global_time_index)
        self.expert_dict[model_id] = {"center": self.expert_centers[-1], "usage": 0, "created": True}
        return len(self.X_list) - 1

    # ----------------------------------------------------------------------
    # Data addition logic
    # ----------------------------------------------------------------------

    def add_point(self, x, y):
        """Add a new (x, y) data point to the appropriate expert."""
        x = np.asarray(x)
        y = np.asarray(y).reshape(-1)

        x_uncat = x.copy()
        y_uncat = y.copy()

        expert_order = self.last_sorted_experts if self.last_sorted_experts is not None else []

        # If no cached order, compute distance to all expert centers
        if self.last_sorted_experts is None:
            print("🔄 Recalculating expert order...")
            outputscale, noise, lengthscale = self.pretrained_params
            sigma_f = np.atleast_1d(outputscale)[0]
            lengthscale = lengthscale if lengthscale.ndim == 1 else lengthscale[:, 0]

            if len(self.expert_centers) == 0:
                expert_order = []
            else:
                dists = [
                    (self.kernel_np(x_uncat[None, :], center[None, :], lengthscale, sigma_f)[0, 0], i)
                    for i, center in enumerate(self.expert_centers)
                ]
                dists.sort()
                expert_order = [i for _, i in dists]

        # Try inserting into one of the nearest experts
        for model in expert_order[: self.nearest_k]:
            self.expert_weights[model] = 1.0
            if model >= len(self.X_list):
                self._create_new_expert(model)

            if self.localCount[model] < self.max_data:
                idx = self.localCount[model]
                self.X_list[model][:, idx] = x_uncat
                self.Y_list[model][:, idx] = y_uncat
                self.localCount[model] += 1

                # Update expert center
                if idx == 0:
                    self.expert_centers[model] = x_uncat
                else:
                    self.expert_centers[model] = (
                        self.expert_centers[model] * idx + x_uncat
                    ) / (idx + 1)

                self.update_param_incremental(x_uncat, y_uncat, model)
                expert_id = self.expert_creation_order[model]
                self.expert_dict[expert_id]["center"] = self.expert_centers[model]
                return

            else:
                # Handle replacement logic
                if not self.replacement:
                    continue
                elif self.drop_counts[model] == 0:
                    d_center = np.linalg.norm(self.expert_centers[model] - x_uncat)
                    stored = self.X_list[model][:, : self.max_data]
                    dists = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                    max_idx = np.argmax(dists)

                    if d_center < dists[max_idx]:
                        # Replace the farthest point
                        x_old = self.X_list[model][:, max_idx].copy()
                        y_old = self.Y_list[model][:, max_idx].copy()

                        self.X_list[model][:, max_idx] = x_uncat
                        self.Y_list[model][:, max_idx] = y_uncat
                        self.expert_centers[model] += (x_uncat - x_old) / self.max_data
                        expert_id = self.expert_creation_order[model]
                        self.expert_dict[expert_id]["center"] = self.expert_centers[model]
                        self.drop_centers[model] = x_old
                        self.drop_counts[model] += 1

                        x_uncat = x_old
                        y_uncat = y_old
                        self.update_param(model)
                        return

                else:
                    stored = self.X_list[model][:, : self.max_data]
                    d_keep = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                    d_drop = np.linalg.norm(stored - self.drop_centers[model][:, None], axis=0)
                    d_new_keep = np.linalg.norm(x_uncat - self.expert_centers[model])
                    d_new_drop = np.linalg.norm(x_uncat - self.drop_centers[model])
                    d_diff = np.concatenate([(d_keep - d_drop), [d_new_keep - d_new_drop]])
                    drop_idx = np.argmax(d_diff)

                    if drop_idx < self.max_data:
                        # Replace an existing point
                        x_old = self.X_list[model][:, drop_idx].copy()
                        y_old = self.Y_list[model][:, drop_idx].copy()

                        self.X_list[model][:, drop_idx] = x_uncat
                        self.Y_list[model][:, drop_idx] = y_uncat

                        self.expert_centers[model] += (x_uncat - x_old) / self.max_data
                        self.drop_centers[model] = (
                            self.drop_centers[model] * self.drop_counts[model] + x_old
                        ) / (self.drop_counts[model] + 1)
                        self.drop_counts[model] += 1

                        x_uncat = x_old
                        y_uncat = y_old
                        self.update_param(model)
                        expert_id = self.expert_creation_order[model]
                        self.expert_dict[expert_id]["center"] = self.expert_centers[model]
                        return

        # Create new expert if all are full
        if self.last_expert_idx is not None:
            model = self._insert_new_expert_near(self.last_expert_idx)
        else:
            model = self._create_new_expert()

        self._insert_into_expert(model, x_uncat, y_uncat)

    # ----------------------------------------------------------------------
    # Expert insertion, replacement, and updates
    # ----------------------------------------------------------------------

    def _insert_new_expert_near(self, near_idx):
        """Insert a new expert near a given expert index."""
        if self.last_x is None or len(self.expert_centers) <= 1:
            return self._create_new_expert()

        left_idx = max(near_idx - 1, 0)
        right_idx = min(near_idx + 1, len(self.expert_centers) - 1)

        outputscale, noise, lengthscale = self.pretrained_params
        sigma_f = np.atleast_1d(outputscale)[0]
        lengthscale = lengthscale if lengthscale.ndim == 1 else lengthscale[:, 0]

        dist_left = self.kernel_np(
            self.last_x[None, :], self.expert_centers[left_idx][None, :], lengthscale, sigma_f
        )[0, 0]
        dist_right = self.kernel_np(
            self.last_x[None, :], self.expert_centers[right_idx][None, :], lengthscale, sigma_f
        )[0, 0]

        insert_after = near_idx if dist_right < dist_left else near_idx - 1
        insert_pos = min(max(insert_after + 1, 0), len(self.expert_centers))

        new_id = max(self.expert_dict.keys(), default=0) + 1

        # Insert new expert data structures
        self.X_list.insert(insert_pos, np.zeros((self.x_dim, self.max_data)))
        self.Y_list.insert(insert_pos, np.zeros((self.y_dim, self.max_data)))
        self.localCount.insert(insert_pos, 0)
        self.expert_centers.insert(insert_pos, np.zeros(self.x_dim))
        self.drop_centers.insert(insert_pos, np.zeros(self.x_dim))
        self.drop_counts.insert(insert_pos, 0)
        self.L_all.insert(insert_pos, np.zeros((self.max_data, self.max_data)))
        self.alpha_all.insert(insert_pos, np.zeros((self.max_data, self.y_dim)))
        self.expert_usage_counts.insert(insert_pos, 0)
        self.expert_creation_order.insert(insert_pos, new_id)
        self.expert_usage_stats.insert(insert_pos, {"used": 0, "considered": 0})
        self.expert_weights.insert(insert_pos, 1.0)

        # Initialize model parameters
        self.init_model_params(new_id)
        self.expert_dict[new_id] = {"center": self.expert_centers[insert_pos], "usage": 0}
        return insert_pos

    def _insert_into_expert(self, model, x, y):
        """Insert (x, y) into an expert."""
        idx = self.localCount[model]
        self.X_list[model][:, idx] = x
        self.Y_list[model][:, idx] = y
        self.localCount[model] += 1
        self.expert_centers[model] = (
            x if idx == 0 else (self.expert_centers[model] * idx + x) / (idx + 1)
        )
        expert_id = self.expert_creation_order[model]
        self.expert_dict[expert_id]["center"] = self.expert_centers[model]
        self.update_param_incremental(x, y, model)

    def _replace_expert(self, idx):
        """Reset and reinitialize a given expert."""
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

    # ----------------------------------------------------------------------
    # Parameter update and prediction
    # ----------------------------------------------------------------------

    def update_param(self, model):
        """Full recomputation of the Cholesky and alpha for one expert."""
        p = 0
        idx = self.localCount[model]
        params = self.model_params[model]
        sigma_f = np.exp(params["log_sigma_f"][p])
        sigma_n = np.exp(params["log_sigma_n"][p])
        lengthscale = (
            np.exp(params["log_lengthscale"]) if self.y_dim == 1 else np.exp(params["log_lengthscale"][:, p])
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

    def predict(self, x_query):
        """Predict the mean and variance at query point x_query."""
        self.last_prediction_cache.clear()

        decay_rate = 0.99
        min_weight_threshold = 1e-3

        if not hasattr(self, "expert_weights"):
            self.expert_weights = [1.0 for _ in self.expert_creation_order]

        self.expert_weights = list(np.array(self.expert_weights) * decay_rate)

        if len(self.expert_centers) == 0:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        raw_ls = self.model_params[0]["log_lengthscale"]
        if raw_ls.ndim == 2:
            lengthscale = np.exp(raw_ls[:, 0])
        else:
            lengthscale = np.exp(raw_ls)

        if self.last_x is not None:
            norm_dist = np.linalg.norm((x_query - self.last_x) / lengthscale)
        else:
            norm_dist = np.inf

        scale = self.timescale
        search_k = int(min(self.max_experts, np.exp(norm_dist / scale)))

        n_experts = len(self.expert_centers)

        if self.last_expert_idx is None:
            candidate_idxs = list(range(n_experts))
        else:
            half_k = search_k // 2
            start = max(0, self.last_expert_idx - half_k)
            end = min(n_experts, self.last_expert_idx + half_k + 1)
            candidate_idxs = list(range(start, end))

        valid_idxs = [idx for idx in candidate_idxs if self.expert_weights[idx] > min_weight_threshold]

        outputscale, noise, lengthscale = self.pretrained_params
        sigma_f = np.atleast_1d(outputscale)[0]
        lengthscale = lengthscale if lengthscale.ndim == 1 else lengthscale[:, 0]
        dists = [
            (
                self.kernel_np(self.expert_centers[idx][:, None], x_query[:, None], lengthscale, sigma_f)[0, 0],
                idx,
            )
            for idx in valid_idxs
        ]

        if not dists:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        dists.sort(reverse=True)
        selected = [idx for _, idx in dists[: self.nearest_k]]
        self.last_sorted_experts = selected

        self.last_x = x_query
        self.last_expert_idx = selected[0]

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
                    np.exp(params["log_lengthscale"][:, p])
                    if self.y_dim > 1
                    else np.exp(params["log_lengthscale"])
                )

                k_star = self.kernel_np(X_snapshot, x_query[:, None], lengthscale, sigma_f).flatten()
                k_xx = sigma_f**2

                mu[p] = np.dot(k_star, alpha[:n_valid, p])
                v = solve_triangular(L[:n_valid, :n_valid], k_star, lower=True)
                var[p] = k_xx - np.sum(v**2)

                if idx not in self.last_prediction_cache:
                    self.last_prediction_cache[idx] = {}
                self.last_prediction_cache[idx][p] = {
                    "k_star": k_star.copy(),
                    "v": v.copy(),
                    "mu_part": mu[p],
                }

            mus.append(mu)
            vars_.append(var)

        mus = np.stack(mus)
        vars_ = np.stack(vars_)
        k = mus.shape[0]
        weights = np.ones((k, 1)) / k

        mu_moe = np.sum(weights * mus, axis=0)
        expected_var = np.sum(weights * vars_, axis=0)
        var_moe = np.clip(expected_var, 1e-6, None)
        return mu_moe, var_moe

    def update_param_incremental(self, x, y, model):
        """Incremental Cholesky and alpha update for a single new point."""
        p = 0
        idx = self.localCount[model]
        if idx == 0:
            return

        params = self.model_params[model]
        sigma_f = np.exp(params["log_sigma_f"][p])
        sigma_n = np.exp(params["log_sigma_n"][p])
        lengthscale = (
            np.exp(params["log_lengthscale"]) if self.y_dim == 1 else np.exp(params["log_lengthscale"][:, p])
        )

        if idx == 1:
            kxx = self.kernel_np(x[:, None], x[:, None], lengthscale, sigma_f)[0, 0] + sigma_n**2
            L = np.sqrt(kxx)
            self.L_all[model][0, 0] = L
            self.alpha_all[model][0, p] = y / L / L
            return

        X_prev = self.X_list[model][:, : idx - 1]
        y_vals = self.Y_list[model][p, :idx]

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
