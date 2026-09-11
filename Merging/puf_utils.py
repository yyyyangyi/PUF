import numpy as np
from scipy.special import xlogy

# 3D scene graph with probabilistic representation
class PUF_SG:
    def __init__(self, num_rel_class, num_obj_class, alpha_prior=0.1,
                 spatial_prior=None, completion_threshold=0.1,
                 obs_strength=1.0, obj_obs_strength=1.0,
                 lambda_birth=0.1, tau_birth=0.5,
                 beta_min=0.05, likelihood_sigma_jsd=1.0,
                 l2_gate=3.0, alpha_min=0.0):
        initial_size = 1000
        self._growth_factor = 2
        self._max_size = initial_size
        self._valid_mask = np.zeros(initial_size, dtype=bool)
        self._classes = np.ndarray(initial_size, dtype=int)
        self._class_alphas = np.zeros((initial_size, num_obj_class), dtype=float)
        self._means = np.ndarray((initial_size, 3), dtype=float)
        self._covs = np.ndarray((initial_size, 3, 3), dtype=float)
        self._rels = np.zeros((initial_size, initial_size, num_rel_class), dtype=float)
        self._rels_observed = np.zeros((initial_size, initial_size), dtype=bool)
        self._pcd = [None] * initial_size

        self.num_rel_class = num_rel_class
        self.num_obj_class = num_obj_class
        self.alpha_prior = alpha_prior
        self.spatial_prior = spatial_prior
        self.completion_threshold = completion_threshold
        self.obs_strength = obs_strength
        self.obj_obs_strength = obj_obs_strength
        self.lambda_birth = lambda_birth
        self.tau_birth = tau_birth
        self.beta_min = beta_min
        self.likelihood_sigma_jsd = likelihood_sigma_jsd
        self.l2_gate = l2_gate
        self.alpha_min = alpha_min

    @property
    def valid_size(self):
        return np.count_nonzero(self._valid_mask)

    @property
    def classes(self):
        return self._classes[self._valid_mask]

    @property
    def means(self):
        return self._means[self._valid_mask]

    @property
    def covs(self):
        return self._covs[self._valid_mask]

    @property
    def rels(self):
        return self._rels[self._valid_mask][:, self._valid_mask]

    @property
    def pcd(self):
        return [self._pcd[i] for i in range(len(self._pcd)) if self._valid_mask[i]]

    @property
    def class_alphas(self):
        return self._class_alphas[self._valid_mask]

    def _expand_if_needed(self, add_size):
        if self.valid_size + add_size <= self._max_size:
            return
        new_size = self._max_size * self._growth_factor + add_size
        add_size = new_size - self._max_size

        self._classes = np.concatenate((self._classes, np.ndarray(add_size, dtype=int)))
        self._class_alphas = np.concatenate((self._class_alphas, np.zeros((add_size, self.num_obj_class), dtype=float)))
        self._means = np.concatenate((self._means, np.ndarray((add_size, 3), dtype=float)))
        self._covs = np.concatenate((self._covs, np.ndarray((add_size, 3, 3), dtype=float)))
        self._rels = np.concatenate((self._rels, np.zeros((add_size, self._max_size, self.num_rel_class), dtype=float)), axis=0)
        self._rels = np.concatenate((self._rels, np.zeros((new_size, add_size, self.num_rel_class), dtype=float)), axis=1)
        self._rels_observed = np.concatenate((self._rels_observed, np.zeros((add_size, self._max_size), dtype=bool)), axis=0)
        self._rels_observed = np.concatenate((self._rels_observed, np.zeros((new_size, add_size), dtype=bool)), axis=1)
        self._pcd.extend([None] * add_size)
        self._valid_mask = np.concatenate((self._valid_mask, np.zeros(add_size, dtype=bool)))
        self._max_size = new_size

    def add(self, new_classes, new_means, new_covs, new_rels, new_rel_classes,
            new_pcds, rel_probs=None, class_probs=None):
        add_size = len(new_classes)
        self._expand_if_needed(add_size)
        avail_indices = np.nonzero(~self._valid_mask)[0][:add_size]
        self._classes[avail_indices] = new_classes
        if class_probs is not None:
            # Soft Dirichlet evidence: initialize alpha from full softmax
            self._class_alphas[avail_indices] = self.obj_obs_strength * class_probs
        else:
            # Hard assignment fallback (GT mode): one-hot
            one_hot = np.zeros((add_size, self.num_obj_class), dtype=float)
            one_hot[np.arange(add_size), new_classes] = 1.0
            self._class_alphas[avail_indices] = one_hot
        self._means[avail_indices] = new_means
        self._covs[avail_indices] = new_covs
        for idx, avail_idx in enumerate(avail_indices):
            self._pcd[avail_idx] = new_pcds[idx]

        if len(new_rels) > 0:
            new_rels_global = avail_indices[new_rels]
            if rel_probs is not None:
                # Soft Dirichlet evidence
                self._rels[new_rels_global[:, 0], new_rels_global[:, 1]] += \
                    self.obs_strength * rel_probs
            else:
                # Hard assignment fallback (GT mode): += 1 one-hot
                self._rels[new_rels_global[:, 0], new_rels_global[:, 1], new_rel_classes] += 1

        self._valid_mask[avail_indices] = True
        return avail_indices

    # ------------------------------------------------------------------
    # Prob. node association functions
    # ------------------------------------------------------------------

    def _compute_likelihood_row(self, obs_idx, candidate_indices):
        """
        Compute L[K] for one observation against K pre-gated candidates.
        """
        # Spatial likelihood via Bhattacharyya coefficient
        hell = self._batched_hellinger_distance(
            self._means[obs_idx], self._covs[obs_idx],
            self._means[candidate_indices], self._covs[candidate_indices],
        )  # (K,)
        L_spatial = 1.0 - hell ** 2

        # Semantic likelihood via JSD on Dirichlet posteriors
        obs_alpha = self._class_alphas[obs_idx]
        obs_s = obs_alpha.sum()
        obs_q = obs_alpha / (obs_s if obs_s > 0 else 1.0)          # (C,)
        cand_alphas = self._class_alphas[candidate_indices]          # (K, C)
        cand_sums = cand_alphas.sum(axis=1)                          # (K,)
        cand_q = cand_alphas / np.where(cand_sums[:, None] > 0, cand_sums[:, None], 1.0)
        mixture = (obs_q + cand_q) / 2                               # (K, C)
        safe_mix = np.where(mixture > 0, mixture, 1.0)
        jsd = 0.5 * (xlogy(obs_q, obs_q / safe_mix).sum(axis=1)
                     + xlogy(cand_q, cand_q / safe_mix).sum(axis=1))  # (K,)
        L_semantic = np.exp(-jsd / self.likelihood_sigma_jsd)

        return L_spatial * L_semantic  # (K,)

    def _jpda_marginals_row(self, L_row):
        """
        Compute JPDA-like marginals for one observation over K gated candidates.
        """
        Z = self.lambda_birth + L_row.sum()
        return L_row / Z, self.lambda_birth / Z

    def fuse(self, update_idx):
        """Merge newly added observation nodes into existing global nodes.

        Args:
            update_idx: indices of newly added observation nodes (from add())
        """
        if len(update_idx) == 0:
            return

        obs_indices = update_idx                           # (N,) newly added
        global_mask = self._valid_mask.copy()
        global_mask[obs_indices] = False
        global_indices = np.nonzero(global_mask)[0]       # (M,) existing nodes

        if len(global_indices) == 0:
            return  # first frame: all obs become new global nodes, already valid

        global_means = self._means[global_indices]         # (M, 3) — cache for L2

        for obs_idx in obs_indices:
            obs_mean = self._means[obs_idx]

            # Phase 1: L2 pre-filter — cheap, no matrix operations
            l2 = np.linalg.norm(global_means - obs_mean[None, :], axis=1)  # (M,)
            gate = l2 < self.l2_gate
            gated_indices = global_indices[gate]

            if len(gated_indices) == 0:
                # No spatially plausible candidate: forced birth, already valid
                continue

            # Phase 2: full likelihood for gated candidates only (K << M)
            L_row = self._compute_likelihood_row(obs_idx, gated_indices)  # (K,)
            beta_d, beta_birth = self._jpda_marginals_row(L_row)          # (K,), scalar

            # Birth condition: birth hypothesis wins (single check, no tau_assign)
            if beta_birth > self.tau_birth:
                continue

            t_star_local = int(np.argmax(beta_d))
            global_t_star = gated_indices[t_star_local]

            # Soft Dirichlet update: spread class evidence weighted by beta
            soft_mask = beta_d >= self.beta_min
            for t_idx in np.nonzero(soft_mask)[0]:
                gt = gated_indices[t_idx]
                w = beta_d[t_idx]
                self._class_alphas[gt] += w * self._class_alphas[obs_idx]
                self._classes[gt] = np.argmax(self._class_alphas[gt])

            # Hard geometry update: only argmax global node
            mean1, cov1 = self._means[obs_idx], self._covs[obs_idx]
            mean2, cov2 = self._means[global_t_star], self._covs[global_t_star]
            n1 = len(self._pcd[obs_idx]) if self._pcd[obs_idx] is not None else 0
            n2 = len(self._pcd[global_t_star]) if self._pcd[global_t_star] is not None else 0
            if n1 == 0 and n2 == 0:
                n1 = n2 = 1
            total = n1 + n2
            diff = mean1 - mean2
            self._means[global_t_star] = (n1 * mean1 + n2 * mean2) / total
            self._covs[global_t_star] = (
                (n1 * cov1 + n2 * cov2) / total
                + n1 * n2 * np.outer(diff, diff) / total ** 2
            )
            if self._pcd[obs_idx] is not None:
                if self._pcd[global_t_star] is not None:
                    self._pcd[global_t_star] = np.concatenate(
                        [self._pcd[global_t_star], self._pcd[obs_idx]]
                    )
                else:
                    self._pcd[global_t_star] = self._pcd[obs_idx].copy()

            # Soft relation transfer, mirrors the soft class update above
            for t_idx in np.nonzero(soft_mask)[0]:
                gt = gated_indices[t_idx]
                w = beta_d[t_idx]
                self._rels[gt, self._valid_mask] += w * self._rels[obs_idx, self._valid_mask]
                self._rels[self._valid_mask, gt] += w * self._rels[self._valid_mask, obs_idx]

            # Invalidate absorbed observation node
            self._valid_mask[obs_idx] = False
            self._classes[obs_idx] = -9999999
            self._class_alphas[obs_idx] = 0.0
            self._means[obs_idx] = np.nan
            self._covs[obs_idx] = np.nan
            self._rels[obs_idx, :] = 0
            self._rels[:, obs_idx] = 0
            self._rels_observed[obs_idx, :] = False
            self._rels_observed[:, obs_idx] = False
            self._pcd[obs_idx] = None

        # Pruning: remove pre-existing global nodes with very low accumulated evidence.
        if self.alpha_min > 0 and len(global_indices) > 0:
            still_valid = self._valid_mask[global_indices]
            alpha_totals = self._class_alphas[global_indices[still_valid]].sum(axis=1)
            to_prune = global_indices[still_valid][alpha_totals < self.alpha_min]
            for idx in to_prune:
                self._valid_mask[idx] = False
                self._classes[idx] = -9999999
                self._class_alphas[idx] = 0.0
                self._means[idx] = np.nan
                self._covs[idx] = np.nan
                self._rels[idx, :] = 0
                self._rels[:, idx] = 0
                self._rels_observed[idx, :] = False
                self._rels_observed[:, idx] = False
                self._pcd[idx] = None

    def _get_prior(self, idx_i, idx_j):
        """Get Dirichlet prior for edge (i, j), using spatial prior if available."""
        if self.spatial_prior is not None:
            return self.spatial_prior.compute(
                self._classes[idx_i], self._classes[idx_j],
                self._means[idx_i], self._means[idx_j]
            )
        else:
            return np.full(self.num_rel_class, self.alpha_prior)

    def complete_relations(self):
        """
        Apply spatial prior to all edges: initialize observed and complete unobserved.
        """
        if self.spatial_prior is None:
            return

        valid_idx = np.nonzero(self._valid_mask)[0]
        for i in valid_idx:
            for j in valid_idx:
                if i == j:
                    continue
                alpha = self._get_prior(i, j)
                if np.sum(self._rels[i, j]) > 0:
                    # Observed edge: add informative prior to existing counts
                    self._rels[i, j, :] += alpha
                else:
                    # Unobserved edge: set prior as prediction if confident
                    if alpha.max() > self.completion_threshold:
                        self._rels[i, j, :] = alpha
                        self._rels_observed[i, j] = True

    def _batched_hellinger_distance(self, mean1, cov1, mean2, cov2):
        """Calculate the Hellinger distance between one and many Gaussian distributions."""
        assert len(mean1.shape) == 1 and len(mean2.shape) == 2
        assert len(cov1.shape) == 2 and len(cov2.shape) == 3
        assert mean1.shape[0] == mean2.shape[1] == 3
        assert cov1.shape[0] == cov1.shape[1] == cov2.shape[1] == cov2.shape[2] == 3
        mean1 = mean1[None, :, None]
        mean2 = mean2[..., None]
        cov1 = cov1[None, ...]
        mean_diff = mean1 - mean2
        cov_mean = (cov1 + cov2) / 2
        cov_mean_inv = np.linalg.inv(cov_mean)
        det_cov_mean = np.linalg.det(cov_mean)
        B_D = (0.125 * mean_diff.transpose(0, 2, 1) @ cov_mean_inv @ mean_diff).flatten() \
            + 0.5 * np.log(det_cov_mean / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
        return np.sqrt(np.clip(1 - np.exp(-B_D), 0, 1))
