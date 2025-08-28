import numpy as np

class UCBGaussian:
    """
    UCB for Gaussian rewards with unknown mean and variance.
    """
    def __init__(self, n_arms: int, n_envs: int, alpha: float = 2.0):
        self.n_arms = n_arms
        self.n_envs = n_envs
        self.alpha = alpha
        self.counts = np.zeros((n_envs, n_arms), dtype=int)    # n_i
        self.sum_rewards = np.zeros((n_envs, n_arms))          # Σx
        self.sum_squares = np.zeros((n_envs, n_arms))          # Σx^2
        self.total_rounds = 0                                # t
        self.warm_up_round = n_arms * 10

    def select_arm(self):
        # warm‐up: pull any arm with fewer than 2 samples
        if self.warm_up_round > 0:
            select_id = self.warm_up_round % self.n_arms
            self.warm_up_round -= 1
            return np.ones(self.n_envs, dtype=int) * select_id

        # compute UCB for each arm
        ucb_values = np.zeros((self.n_envs, self.n_arms))
        for i in range(self.n_arms):
            n_i = self.counts[:, i]
            mean = self.sum_rewards[:, i] / n_i

            var = (self.sum_squares[:, i]
                   - (self.sum_rewards[:, i]**2) / n_i) / (n_i - 1)
            var = np.maximum(var, 1e-6)
            bonus = np.sqrt(self.alpha * var * np.log(self.total_rounds) / n_i)
            ucb_values[:, i] = mean + bonus

        return np.argmax(ucb_values, axis=1)

    def update(self, arm, reward):
        """
        After pulling `arm` and observing `reward`, update statistics.
        """
        self.total_rounds += 1
        idx = np.arange(self.n_envs)
        self.counts[idx, arm] += 1
        self.sum_rewards[idx, arm] += reward.squeeze()
        self.sum_squares[idx, arm] += reward.squeeze()**2
