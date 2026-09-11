import numpy as np


class SpatialPrior:
    """
    Class-conditional and spatial relation prior for 3D scene graphs.
    """

    def __init__(self, num_obj_classes=20, num_rel_classes=7,
                 class_prior_path=None, alpha_strength=1.0, sigma_d=1.0):
        """
        Args:
            num_obj_classes: number of object classes
            num_rel_classes: number of relation classes
            class_prior_path: path to precomputed P_class .npy file.
                              If None, uses uniform class prior (training-free mode).
            alpha_strength: overall prior strength (scales output Dirichlet pseudo-counts)
            sigma_d: distance scale for spatial prior (meters)
        """
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes = num_rel_classes
        self.alpha_strength = alpha_strength
        self.sigma_d = sigma_d

        if class_prior_path is not None:
            data = np.load(class_prior_path)
            self._class_prior = data['P_class']
            self._exist_prior = data['P_exist']
            assert self._class_prior.shape == (num_obj_classes, num_obj_classes, num_rel_classes), \
                f"Class prior shape mismatch: {self._class_prior.shape} vs expected ({num_obj_classes}, {num_obj_classes}, {num_rel_classes})"
        else:
            # Uniform class prior (training-free mode), no existence gate
            self._class_prior = np.ones((num_obj_classes, num_obj_classes, num_rel_classes)) / num_rel_classes
            self._exist_prior = None
            print("Using uniform class prior (training-free mode)")

    def compute(self, class_i, class_j, mean_i, mean_j):
        """Compute informative Dirichlet prior for edge (i, j).

        Args:
            class_i: int, subject object class index
            class_j: int, object object class index
            mean_i: (3,) subject 3D position
            mean_j: (3,) object 3D position

        Returns:
            (num_rel_classes,) Dirichlet alpha values
        """
        P_class = self._class_prior[class_i, class_j]
        P_spatial = self._compute_spatial(mean_i, mean_j)
        P_combined = P_class * P_spatial
        total = P_combined.sum()
        if total > 1e-12:
            P_combined = P_combined / total
        else:
            P_combined = np.ones(self.num_rel_classes) / self.num_rel_classes

        if self._exist_prior is not None:
            exist_prob = self._exist_prior[class_i, class_j]
        else:
            exist_prob = 1.0

        return P_combined * self.alpha_strength * exist_prob

    def compute_batched(self, classes, means, pairs):
        """Compute priors for multiple node pairs at once.

        Args:
            classes: (N,) int array of class indices
            means: (N, 3) float array of 3D positions
            pairs: (M, 2) int array of (i, j) node index pairs

        Returns:
            (M, num_rel_classes) Dirichlet alpha values per pair
        """
        M = len(pairs)
        alphas = np.zeros((M, self.num_rel_classes), dtype=float)
        for k in range(M):
            i, j = pairs[k]
            alphas[k] = self.compute(classes[i], classes[j], means[i], means[j])
        return alphas

    def _compute_spatial(self, mean_i, mean_j):
        """Compute spatial likelihood for each relation type.

        Relation indices (ScanNet):
            0: attached to, 1: build in, 2: connected to,
            3: hanging on, 4: part of, 5: standing on, 6: supported by

        Args:
            mean_i: (3,) subject 3D position
            mean_j: (3,) object 3D position

        Returns:
            (num_rel_classes,) unnormalized spatial scores
        """
        delta = mean_j - mean_i
        distance = np.linalg.norm(delta) + 1e-6
        vertical = delta[2]  # positive = j above i (z-up convention)
        horizontal = np.sqrt(delta[0] ** 2 + delta[1] ** 2)

        proximity = np.exp(-distance / self.sigma_d)

        scores = np.zeros(self.num_rel_classes)

        scores[0] = proximity
        scores[1] = proximity
        scores[2] = proximity
        vert_down = max(0.0, -vertical)
        scores[3] = proximity * (1.0 + vert_down / self.sigma_d) * np.exp(-horizontal / self.sigma_d)
        scores[4] = proximity
        vert_up = max(0.0, vertical)
        scores[5] = proximity * (1.0 + vert_up / self.sigma_d) * np.exp(-horizontal / self.sigma_d)
        scores[6] = proximity * (1.0 + vert_down / self.sigma_d)

        # Normalize to probability
        total = scores.sum()
        if total > 1e-12:
            scores = scores / total
        else:
            scores = np.ones(self.num_rel_classes) / self.num_rel_classes

        return scores
