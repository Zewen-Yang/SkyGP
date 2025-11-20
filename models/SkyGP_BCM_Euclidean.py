import numpy as np
from scipy.linalg import solve_triangular


def sigmoid(x, temp=10.0):
    """Sigmoid with temperature scaling (controls growth speed)."""
    return 1 / (1 + np.exp(-x / temp))


class SkyGP_BCM_Euclidean:
    """
    Bayesian Committee Machine (BCM) over local GP experts.
    Expert selection uses Euclidean distance between expert centers and query.
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

        # Runtime state
        self.last_sorted_experts = None            # Cached expert order
        self.last_prediction_cache = {}            # Cache: k_star, v, mu_part per expert
        self.expert_usage_counts = []              # Per-expert usage count
        self.replacement = replacement             # Enable replacement policy when full
        self.expert_creation_order = []            # Expert IDs in creation order
        self.expert_usage_stats = []               # {'used': ..., 'considered': ...} per expert
        self.expert_weights = []                   # Decayed weight per expert
        self.last_x = None                         # Last data/query for distance calc
        self.last_expert_idx = None                # Index of last-used expert
        self.expert_dict = {}                      # Metadata per expert (hash map)
        self.expert_id_counter = 0                 # Sequential expert id

    # --------------------------------------------------------------------- #
    # Core GP ops
    # --------------------------------------------------------------------- #

    def kernel_np(self, X1, X2, lengthscale, sigma_f):
        """Squared exponential (RBF) kernel."""
        X1_scaled = X1 / lengthscale[:, None]
        X2_scaled = X2 / lengthscale[:, None]
        dists = np.sum((X1_scaled[:, :, None] - X2_scaled[:, None, :]) ** 2, axis=0)
        return sigma_f**2 * np.exp(-0.5 * dists)

    def init_model_params(self, model_id, pretrained_params=None):
        """Initialize hyperparameters (use pretrained if available)."""
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
            # Default priors
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

    # --------------------------------------------------------------------- #
    # Expert management
    # --------------------------------------------------------------------- #

    def _create_new_expert(self, model_id=None):
        """Create and register a new expert, returning its list index."""
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
        self.expert_dict[model_id] = {"center": self.expert_centers[-1], "usage": 0, "created": True}

        return len(self.X_list) - 1

    def _replace_expert(self, idx):
        """Hard-reset an existing expert slot."""
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

    # --------------------------------------------------------------------- #
    # Data ingestion
    # --------------------------------------------------------------------- #

    def add_point(self, x, y):
        """Add one (x, y) sample to the nearest feasible expert; create/insert if needed."""
        x = np.asarray(x)
        y = np.asarray(y).reshape(-1)

        x_uncat = x.copy()
        y_uncat = y.copy()

        # Use cached order if available; otherwise recompute by kernel similarity to centers
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

        # Try to insert into the nearest_k experts
        for model in expert_order[:self.nearest_k]:
            if model >= len(self.X_list):
                self._create_new_expert(model)

            # If capacity available: insert directly
            if self.localCount[model] < self.max_data:
                idx = self.localCount[model]
                self.X_list[model][:, idx] = x_uncat
                self.Y_list[model][:, idx] = y_uncat
                self.localCount[model] += 1

                # Update running center
                if idx == 0:
                    self.expert_centers[model] = x_uncat
                else:
                    self.expert_centers[model] = (self.expert_centers[model] * idx + x_uncat) / (idx + 1)

                self.update_param_incremental(x_uncat, y_uncat, model)
                expert_id = self.expert_creation_order[model]
                self.expert_dict[expert_id]["center"] = self.expert_centers[model]
                return

            # Otherwise consider replacement policy
            if not self.replacement:
                continue

            # First replacement (MASGP-like): compare distances from center
            if self.drop_counts[model] == 0:
                d_center = np.linalg.norm(self.expert_centers[model] - x_uncat)
                stored = self.X_list[model][:, :self.max_data]
                dists = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                max_idx = np.argmax(dists)

                if d_center < dists[max_idx]:
                    # Replace the farthest-from-center sample
                    x_old = self.X_list[model][:, max_idx].copy()
                    y_old = self.Y_list[model][:, max_idx].copy()

                    self.X_list[model][:, max_idx] = x_uncat
                    self.Y_list[model][:, max_idx] = y_uncat

                    # Update center incrementally
                    self.expert_centers[model] += (x_uncat - x_old) / self.max_data
                    expert_id = self.expert_creation_order[model]
                    self.expert_dict[expert_id]["center"] = self.expert_centers[model]

                    # Track dropped sample "center"
                    self.drop_centers[model] = x_old
                    self.drop_counts[model] += 1

                    # Push old sample forward to continue routing
                    x_uncat = x_old
                    y_uncat = y_old
                    self.update_param(model)
                    self.expert_weights[model] = 1.0  # Reset weight
                    return

            else:
                # Subsequent replacements: discriminate between keep/drop centers
                stored = self.X_list[model][:, :self.max_data]
                d_keep = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                d_drop = np.linalg.norm(stored - self.drop_centers[model][:, None], axis=0)
                d_new_keep = np.linalg.norm(x_uncat - self.expert_centers[model])
                d_new_drop = np.linalg.norm(x_uncat - self.drop_centers[model])

                d_diff = np.concatenate([(d_keep - d_drop), [d_new_keep - d_new_drop]])
                drop_idx = np.argmax(d_diff)

                if drop_idx < self.max_data:
                    # Replace existing point
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
                    self.expert_weights[model] = 1.0  # Reset weight
                    return

        # If none of the nearest experts can take it, insert a new expert near the last used one
        if self.last_expert_idx is not None:
            model = self._insert_new_expert_near(self.last_expert_idx)
        else:
            model = self._create_new_expert()

        self._insert_into_expert(model, x_uncat, y_uncat)

    def _insert_new_expert_near(self, near_idx):
        """
        Insert a new expert near the last used expert (left/right) using Euclidean distance.
        """
        if self.last_x is None or len(self.expert_centers) <= 1:
            return self._create_new_expert()

        left_idx = max(near_idx - 1, 0)
        right_idx = min(near_idx + 1, len(self.expert_centers) - 1)

        # Compare Euclidean distances to decide insertion side (slight right bias)
        dist_left = np.linalg.norm(self.last_x - self.expert_centers[left_idx])
        dist_right = np.linalg.norm(self.last_x - self.expert_centers[right_idx])

        insert_after = near_idx if dist_right < dist_left else near_idx - 1
        insert_pos = min(max(insert_after + 1, 0), len(self.expert_centers))

        # New expert ID
        new_id = max(self.expert_dict.keys(), default=0) + 1

        # Insert synchronized structures
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

        # Initialize params and metadata
        self.init_model_params(new_id)
        self.expert_dict[new_id] = {"center": self.expert_centers[insert_pos], "usage": 0}

        return insert_pos

    def _insert_into_expert(self, model, x, y):
        """Insert (x, y) into a specific expert and update factors incrementally."""
        idx = self.localCount[model]
        self.X_list[model][:, idx] = x
        self.Y_list[model][:, idx] = y
        self.localCount[model] += 1
        self.expert_centers[model] = x if idx == 0 else (self.expert_centers[model] * idx + x) / (idx + 1)
        expert_id = self.expert_creation_order[model]
        self.expert_dict[expert_id]["center"] = self.expert_centers[model]
        self.update_param_incremental(x, y, model)

    # --------------------------------------------------------------------- #
    # Full and incremental parameter updates
    # --------------------------------------------------------------------- #

    def update_param(self, model):
        """Full recomputation of Cholesky factor L and alpha for one expert."""
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
        """Rank-1 incremental update of L and alpha when appending a single point."""
        p = 0
        idx = self.localCount[model]
        if idx == 0:
            return

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

        # Use cached k_star and v from last prediction if they match the current size
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

    # --------------------------------------------------------------------- #
    # Prediction (BCM fusion)
    # --------------------------------------------------------------------- #

    def predict(self, x_query):
        """
        Predict mean/variance using BCM over nearest_k experts selected by Euclidean distance.
        Weights are precision-based; standard BCM correction is applied.
        """
        self.last_prediction_cache.clear()

        # Exponential decay on expert weights
        decay_rate = 0.999
        min_weight_threshold = 1e-3

        if not hasattr(self, "expert_weights"):
            self.expert_weights = [1.0 for _ in self.expert_creation_order]
        self.expert_weights = [w * decay_rate for w in self.expert_weights]

        if len(self.expert_centers) == 0:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        # Lengthscale from model 0 for normalization in search window
        raw_ls = self.model_params[0]["log_lengthscale"]
        lengthscale = np.exp(raw_ls[:, 0]) if raw_ls.ndim == 2 else np.exp(raw_ls)

        if self.last_x is not None:
            norm_dist = np.linalg.norm((x_query - self.last_x) / lengthscale)
        else:
            norm_dist = np.inf

        # Exponential growth of search window with distance from last query
        base_k = 1           # Minimum number of experts to consider
        growth_rate = 2.0    # (Unused constant kept for clarity)
        scale = 0.02         # Scale to temper growth
        search_k = int(min(self.max_experts, base_k * np.exp(norm_dist / scale)))

        n_experts = len(self.expert_centers)
        if self.last_expert_idx is None:
            candidate_idxs = list(range(n_experts))
        else:
            half_k = search_k // 2
            start = max(0, self.last_expert_idx - half_k)
            end = min(n_experts, self.last_expert_idx + half_k + 1)
            candidate_idxs = list(range(start, end))

        # Filter by minimum weight
        valid_idxs = [idx for idx in candidate_idxs if self.expert_weights[idx] > min_weight_threshold]

        if self.pretrained_params is None:
            # If no pretrained params, we still rank by Euclidean distance
            pass

        # Rank by Euclidean distance to centers (ascending)
        dists = [(np.linalg.norm(self.expert_centers[idx] - x_query), idx) for idx in valid_idxs]
        if not dists:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        dists.sort()
        selected = [idx for _, idx in dists[: self.nearest_k]]
        self.last_sorted_experts = selected
        self.last_x = x_query
        self.last_expert_idx = selected[0]

        # Per-expert GP predictions
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

                # Cache for possible incremental update
                if idx not in self.last_prediction_cache:
                    self.last_prediction_cache[idx] = {}
                self.last_prediction_cache[idx][p] = {
                    "k_star": k_star.copy(),
                    "v": v.copy(),
                    "mu_part": mu[p],
                }

            mus.append(mu)
            vars_.append(var)

        mus = np.stack(mus)          # (k, y_dim)
        vars_ = np.stack(vars_)      # (k, y_dim)

        # BCM fusion
        inv_vars = 1.0 / (vars_ + 1e-9)       # Precision weights
        sigma0_sq = np.exp(self.model_params[selected[0]]["log_sigma_f"][0]) ** 2

        mu_weighted = np.sum(inv_vars * mus, axis=0)
        denom = np.sum(inv_vars, axis=0)

        # BCM correction: subtract prior counted (k-1) times
        denom_corr = denom - (len(mus) - 1) / sigma0_sq
        mu_bcm = mu_weighted / (denom_corr + 1e-9)
        var_bcm = 1.0 / (denom_corr + 1e-9)

        return mu_bcm, var_bcm
