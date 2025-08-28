import heapq
import numpy as np
from typing import Optional, Sequence, List

class _KSmallestTracker:
    """
    Maintains the sum of the K smallest elements in a multiset with O(log n) inserts.
    Internals: max-heap `lower` holds K smallest (as negatives), min-heap `upper` the rest.
    """
    __slots__ = ("K", "lower", "upper", "sum_lower", "n")

    def __init__(self, K: int = 0):
        self.K = max(0, int(K))
        self.lower: List[float] = []  # store as NEGATIVES (max-heap behavior)
        self.upper: List[float] = []  # min-heap
        self.sum_lower: float = 0.0
        self.n: int = 0

    def _rebalance_to_K(self):
        # Make |lower| == K by moving items between heaps
        while len(self.lower) < self.K and self.upper:
            x = heapq.heappop(self.upper)         # smallest in upper
            heapq.heappush(self.lower, -x)
            self.sum_lower += x
        while len(self.lower) > self.K:
            x = -heapq.heappop(self.lower)        # largest among K-smallest
            self.sum_lower -= x
            heapq.heappush(self.upper, x)

    def set_K(self, K_new: int):
        self.K = max(0, int(K_new))
        self._rebalance_to_K()

    def insert(self, x: float):
        self.n += 1
        # If we still have slack in lower, push directly there.
        if len(self.lower) < self.K:
            heapq.heappush(self.lower, -x)
            self.sum_lower += x
            return

        # Otherwise route by comparing to largest among current K-smallest
        if self.lower and x < -self.lower[0]:
            # x belongs to 'lower'; evict current max from 'lower' to 'upper'
            largest_lower = -heapq.heapreplace(self.lower, -x)
            self.sum_lower += x - largest_lower
            heapq.heappush(self.upper, largest_lower)
        else:
            heapq.heappush(self.upper, x)

    def min_upper(self) -> Optional[float]:
        """Returns the smallest element NOT counted in the K sum (None if all are counted)."""
        return self.upper[0] if self.upper else None

    def sumK(self) -> float:
        return self.sum_lower

    def size(self) -> int:
        return self.n


class _ArmState:
    """
    Per-(env, arm) state: keeps two trackers:
      S_m    : sum of the m = ceil(ε n) smallest
      S_mk   : sum of the m+k smallest, where k = floor(α n)
    Also tracks total sum, cached ucb, and current n.
    """
    __slots__ = ("alpha","delta","U","n","sum_all","S_m","S_mk","m","k","ucb_cache")

    def __init__(self, alpha: float, delta: float, U: float):
        self.alpha = alpha
        self.delta = delta
        self.U = float(U)
        self.n = 0
        self.sum_all = 0.0
        self.S_m = _KSmallestTracker(0)
        self.S_mk = _KSmallestTracker(0)
        self.m = 0
        self.k = 0
        self.ucb_cache = float('inf')  # unseen arms = +inf

    @staticmethod
    def _dkw_eps(n: int, delta: float) -> float:
        return 0.0 if n <= 0 else float(np.sqrt(np.log(2.0/delta)/(2.0*n)))

    def _recompute_targets(self):
        """Update m,k and trackers' K after n changed."""
        if self.n == 0:
            self.m, self.k = 0, 0
            self.S_m.set_K(0)
            self.S_mk.set_K(0)
            return
        eps = self._dkw_eps(self.n, self.delta)
        m_new = int(np.ceil(eps * self.n))
        k_new = int(np.floor(self.alpha * self.n))
        # Bound to [0, n]
        m_new = max(0, min(m_new, self.n))
        mk_new = max(0, min(m_new + k_new, self.n))
        self.m, self.k = m_new, k_new
        self.S_m.set_K(m_new)
        self.S_mk.set_K(mk_new)

    def _recompute_ucb_cache(self):
        """Compute optimistic CVaR_α UCB using the maintained order stats."""
        if self.n == 0:
            self.ucb_cache = float('inf')
            return
        r = self.alpha * self.n            # α n
        k = int(np.floor(r))
        eta = r - k

        # Number of non-U values left after bumping m smallest to U
        q = self.n - self.m

        S_m = self.S_m.sumK()
        S_mk = self.S_mk.sumK()
        next_after_mk = self.S_mk.min_upper()

        if r <= q:
            # We can fill the α-tail entirely from the non-U values
            # numerator = sum_{i=m+1}^{m+k} + η * y_{(m+k+1)}
            numer = S_mk - S_m
            if eta > 0.0:
                # If we exactly hit the boundary (m+k == n), upper may be empty; fallback to U (it won't matter if eta=0)
                y_next = next_after_mk if next_after_mk is not None else self.U
                numer += eta * y_next
        else:
            # α-tail spills into the U's: include all q non-U values plus leftover mass on U
            numer = (self.sum_all - S_m) + (r - q) * self.U

        self.ucb_cache = numer / r if r > 0 else self.U  # α>0 always; safeguard

    def update(self, x: float):
        """Insert a new reward and refresh cache."""
        self.n += 1
        self.sum_all += float(x)
        # Insert with *previous* K; then adjust K to new (m,k) and rebalance
        self.S_m.insert(x)
        self.S_mk.insert(x)
        self._recompute_targets()
        self._recompute_ucb_cache()


class CVAR_UCB:
    """
    Multi-environment K-armed bandit using DKW-optimistic CVaR_α (rewards) as UCB, with O(log n) updates.

    - Rewards must be upper bounded by U
    - Each (env, arm) keeps online order-stats to avoid sorting on every update.
    - `select_arms()` is O(n_env * n_arms) but only does array argmax over cached UCBs.
    """

    def __init__(
        self,
        n_arms: int,
        n_env: int,
        alpha: float = 0.3,
        delta: float = 0.05,
        U: float = 20,
        seed: Optional[int] = None,
    ):
        if n_env <= 0 or n_arms <= 0:
            raise ValueError("n_env and n_arms must be >= 1")
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0,1]")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0,1)")
        self.n_env = n_env
        self.n_arms = n_arms
        self.alpha = alpha
        self.delta = delta
        self.U = float(U)
        self.rng = np.random.default_rng(seed)

        # Per-env array of ArmState
        self._arms: List[List[_ArmState]] = [
            [_ArmState(alpha, delta, self.U) for _ in range(n_arms)]
            for _ in range(n_env)
        ]

    def select_arm(self) -> np.ndarray:
        """Return (n_env,) arm indices with maximal optimistic CVaR_α (UCB)."""
        choices = np.zeros(self.n_env, dtype=int)
        # small noise to break ties deterministically/randomly
        noise = self.rng.random((self.n_env, self.n_arms)) * 1e-12
        for e in range(self.n_env):
            ucbs = np.array([self._arms[e][a].ucb_cache for a in range(self.n_arms)])
            a = int(np.argmax(ucbs + noise[e]))
            choices[e] = a
        return choices

    def update(self, arms: Sequence[int], rewards: Sequence[float]) -> None:
        """Update per-env statistics; both arrays must be shape (n_env,)."""
        arms = np.asarray(arms, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        if arms.shape != (self.n_env,) or rewards.shape != (self.n_env,):
            raise ValueError(f"arm_ids and rewards must each have shape ({self.n_env},)")
        for e in range(self.n_env):
            a = int(arms[e])
            if not (0 <= a < self.n_arms):
                raise IndexError(f"arm index {a} out of range for env {e}")
            r = float(rewards[e])
            # Enforce support bound (optional safety)
            if r > self.U:
                r = self.U
            self._arms[e][a].update(r)

    # Optional: diagnostics
    def counts(self) -> np.ndarray:
        C = np.zeros((self.n_env, self.n_arms), dtype=int)
        for e in range(self.n_env):
            for a in range(self.n_arms):
                C[e, a] = self._arms[e][a].n
        return C

    def ucbs_matrix(self) -> np.ndarray:
        M = np.zeros((self.n_env, self.n_arms), dtype=float)
        for e in range(self.n_env):
            for a in range(self.n_arms):
                M[e, a] = self._arms[e][a].ucb_cache
        return M
