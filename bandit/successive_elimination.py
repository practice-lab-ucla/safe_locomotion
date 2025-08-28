import numpy as np

class SuccessiveElimination:
    """
    Successive Elimination for Gaussian rewards with unknown mean/variance.

    The class mirrors the API of `UCBGaussian`:
      - `select_arm()`  -> np.ndarray of shape (n_envs,)
      - `update(arm, reward)` updates internal statistics
    """
    def __init__(self, n_arms: int, n_envs: int, alpha: float = 0.5):
        self.n_arms   = n_arms
        self.n_envs   = n_envs
        self.alpha    = alpha               # confidence scale
        self.counts        = np.zeros((n_envs, n_arms), dtype=int)
        self.sum_rewards   = np.zeros((n_envs, n_arms))
        self.sum_squares   = np.zeros((n_envs, n_arms))
        self.total_rounds  = 0

        # Each env keeps its own active‑arm mask
        self.active = np.ones((n_envs, n_arms), dtype=bool)

    # ------------------------------------------------------------
    def _confidence_radius(self, var, n_i):
        """
        Empirical-Bernstein radius; falls back to Hoeffding when n_i<2.
        """
        # avoid divide‑by‑zero
        n_i = np.maximum(n_i, 1)
        rad = np.sqrt(self.alpha * var * np.log(self.total_rounds+1) / n_i)
        # tie‑in a Hoeffding term to stay valid when var≈0
        rad += np.sqrt(self.alpha * np.log(self.total_rounds+1) / (2*n_i))
        return rad

    # ------------------------------------------------------------
    def select_arm(self):
        """
        For every environment, pick the *least-sampled* remaining arm.
        This keeps sampling roughly uniform over the active set.
        Returns: np.ndarray [n_envs] with selected arm indices.
        """
        choice = np.zeros(self.n_envs, dtype=int)
        for e in range(self.n_envs):
            # active arms and their pull counts
            active_idx = np.where(self.active[e])[0]
            c = self.counts[e, active_idx]
            # pick arm with minimum pulls (ties broken by idx order)
            choice[e] = active_idx[np.argmin(c)]
        return choice

    # ------------------------------------------------------------
    def update(self, arm, reward):
        """
        After pulling `arm` (vector of length n_envs) and observing `reward`
        (same shape), update stats **and run an elimination test**.
        """
        self.total_rounds += 1
        idx = np.arange(self.n_envs)

        self.counts[idx, arm]      += 1
        self.sum_rewards[idx, arm] += reward.squeeze()
        self.sum_squares[idx, arm] += reward.squeeze()**2

        # ---------- elimination step ----------
        for e in range(self.n_envs):
            active_idx = np.where(self.active[e])[0]
            if len(active_idx) <= 1:           # nothing to eliminate
                continue

            n_i   = self.counts[e, active_idx]
            means = self.sum_rewards[e, active_idx] / np.maximum(n_i, 1)

            # unbiased sample variance; clip to keep numerically positive
            var = (self.sum_squares[e, active_idx] -
                   (self.sum_rewards[e, active_idx]**2) / np.maximum(n_i, 1))
            var /= np.maximum(n_i-1, 1)
            var  = np.maximum(var, 1e-6)

            rad   = self._confidence_radius(var, n_i)

            ucb = means + rad
            lcb = means - rad

            best_lcb = np.max(lcb)

            # eliminate arms whose UCB is *strictly* below best LCB
            to_drop = active_idx[ucb < best_lcb]
            self.active[e, to_drop] = False
