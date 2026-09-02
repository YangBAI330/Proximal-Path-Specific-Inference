import numpy as np


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_uniform_disjoint(low, high, size):
    signs = 2.0 * (np.random.choice(2, size=size) - 0.5)
    vals = np.random.uniform(low=low, high=high, size=size)
    return signs * vals


def sample_fixed_midpoint(low, high, size, scale=1.0):
    return np.full(size, scale * 0.5 * (low + high), dtype=float)


def centered_mean_square(*arrays):
    parts = []
    for arr in arrays:
        sq = arr * arr
        parts.append(sq.mean(axis=1, keepdims=True) - sq.mean())
    return sum(parts) / float(len(parts))


def repeat_to_width(x, width):
    return np.repeat(x, width, axis=1)


class ExtendedLinearDGP:
    """Linear DGP used by the original simulation code."""

    def __init__(
        self,
        udim,
        xdim,
        zdim,
        wdim,
        ddim,
        mdim,
        ydim=1,
        l=0.5,
        u=1.0,
        var=0.5,
        proxy_strength=1.0,
        proxy_noise=1.0,
        treatment_proxy_strength=1.0,
        outcome_proxy_strength=1.0,
        fixed_weights=False,
        fixed_weight_scale=1.0,
        proxy_square_strength=0.0,
        outcome_square_strength=0.0,
        nonnegative=False,
        azwy_nonnegative=False,
        seed=0,
    ):
        np.random.seed(seed)

        self.udim = udim
        self.xdim = xdim
        self.zdim = zdim
        self.wdim = wdim
        self.ddim = ddim
        self.mdim = mdim
        self.ydim = ydim
        self.proxy_strength = proxy_strength
        self.proxy_noise = proxy_noise
        self.treatment_proxy_strength = treatment_proxy_strength
        self.outcome_proxy_strength = outcome_proxy_strength
        self.fixed_weights = fixed_weights
        self.fixed_weight_scale = fixed_weight_scale
        self.proxy_square_strength = proxy_square_strength
        self.outcome_square_strength = outcome_square_strength

        if fixed_weights:
            sampler = lambda low, high, size: sample_fixed_midpoint(low, high, size, fixed_weight_scale)
            azwy_sampler = sampler
        else:
            sampler = np.random.uniform if nonnegative else sample_uniform_disjoint
            azwy_sampler = np.random.uniform if azwy_nonnegative else sampler

        self.Wux = sampler(low=l, high=u, size=(udim, xdim)) / np.sqrt(udim)
        self.Wuz = proxy_strength * sampler(low=l, high=u, size=(udim, zdim)) / np.sqrt(udim)
        self.Wxz = sampler(low=l, high=u, size=(xdim, zdim)) / np.sqrt(xdim)
        self.Wuw = proxy_strength * sampler(low=l, high=u, size=(udim, wdim)) / np.sqrt(udim)
        self.Wxw = sampler(low=l, high=u, size=(xdim, wdim)) / np.sqrt(xdim)

        self.Wua = sampler(low=0.4 * l, high=0.4 * u, size=(udim, 1)) / np.sqrt(udim)
        self.Wxa = sampler(low=0.4 * l, high=0.4 * u, size=(xdim, 1)) / np.sqrt(xdim)
        self.Wza = treatment_proxy_strength * sampler(low=0.4 * l, high=0.4 * u, size=(zdim, 1)) / np.sqrt(zdim)

        self.Wud = sampler(low=l, high=u, size=(udim, ddim)) / np.sqrt(udim)
        self.Wxd = sampler(low=l, high=u, size=(xdim, ddim)) / np.sqrt(xdim)
        self.Wad = sampler(low=l, high=u, size=(1, ddim))

        self.Wum = sampler(low=l, high=u, size=(udim, mdim)) / np.sqrt(udim)
        self.Wxm = sampler(low=l, high=u, size=(xdim, mdim)) / np.sqrt(xdim)
        self.Wam = sampler(low=l, high=u, size=(1, mdim))
        self.Wdm = sampler(low=l, high=u, size=(ddim, mdim)) / np.sqrt(ddim)

        self.Wuy = 2.0 * sampler(low=l, high=u, size=(udim, ydim)) / np.sqrt(udim)
        self.Wxy = 2.0 * sampler(low=l, high=u, size=(xdim, ydim)) / np.sqrt(xdim)
        self.Wwy = outcome_proxy_strength * azwy_sampler(low=l, high=u, size=(wdim, ydim)) / np.sqrt(wdim)
        self.Way = sampler(low=l, high=u, size=(1, ydim))
        self.Wdy = sampler(low=l, high=u, size=(ddim, ydim)) / np.sqrt(ddim)
        self.Wmy = sampler(low=l, high=u, size=(mdim, ydim)) / np.sqrt(mdim)

        self.ucov = np.eye(udim)
        self.xcov = var * np.eye(xdim)
        self.zcov = var * proxy_noise * np.eye(zdim)
        self.wcov = var * proxy_noise * np.eye(wdim)
        self.acov = var
        self.dcov = var * np.eye(ddim)
        self.mcov = var * np.eye(mdim)
        self.ycov = var * np.eye(ydim)

    def sample_dataset(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)

        eps_u = np.random.multivariate_normal(np.zeros(self.udim), self.ucov, n)
        eps_x = np.random.multivariate_normal(np.zeros(self.xdim), self.xcov, n)
        eps_z = np.random.multivariate_normal(np.zeros(self.zdim), self.zcov, n)
        eps_w = np.random.multivariate_normal(np.zeros(self.wdim), self.wcov, n)
        eps_d = np.random.multivariate_normal(np.zeros(self.ddim), self.dcov, n)
        eps_m = np.random.multivariate_normal(np.zeros(self.mdim), self.mcov, n)
        eps_y = np.random.multivariate_normal(np.zeros(self.ydim), self.ycov, n)

        U = eps_u
        X = U @ self.Wux + eps_x
        Z = U @ self.Wuz + X @ self.Wxz + eps_z
        W = U @ self.Wuw + X @ self.Wxw + eps_w
        if self.proxy_square_strength != 0:
            proxy_square = self.proxy_square_strength * centered_mean_square(U, X)
            Z = Z + repeat_to_width(proxy_square, self.zdim)
            W = W + repeat_to_width(proxy_square, self.wdim)

        a_noise = np.random.normal(0.0, self.acov, n).reshape(-1, 1)
        a_logit = U @ self.Wua + X @ self.Wxa + Z @ self.Wza + a_noise
        A_probs = _expit(a_logit).flatten()
        A = np.random.binomial(1, A_probs).reshape(-1, 1)

        D = U @ self.Wud + X @ self.Wxd + A @ self.Wad + eps_d
        M = U @ self.Wum + X @ self.Wxm + A @ self.Wam + D @ self.Wdm + eps_m
        Y = (
            U @ self.Wuy
            + X @ self.Wxy
            + W @ self.Wwy
            + A @ self.Way
            + D @ self.Wdy
            + M @ self.Wmy
            + self._outcome_square(W, D, M, X)
            + eps_y
        )

        return {
            "u": U,
            "x": X,
            "z": Z,
            "w": W,
            "a": A,
            "d": D,
            "m": M,
            "y": Y,
            "a_p": A_probs.reshape(-1, 1),
        }

    def _outcome_square(self, W, D, M, X):
        if self.outcome_square_strength == 0:
            return np.zeros((W.shape[0], self.ydim))
        basis = centered_mean_square(W, D, M, X)
        return self.outcome_square_strength * repeat_to_width(basis, self.ydim)

    def true_psi(self, data):
        U = data["u"]
        X = data["x"]
        W = data["w"]
        n = X.shape[0]

        A1 = np.ones((n, 1))
        A0 = np.zeros((n, 1))
        D_A1 = U @ self.Wud + X @ self.Wxd + A1 @ self.Wad
        M_A0_D_A1 = U @ self.Wum + X @ self.Wxm + A0 @ self.Wam + D_A1 @ self.Wdm
        Y = (
            U @ self.Wuy
            + X @ self.Wxy
            + W @ self.Wwy
            + A1 @ self.Way
            + D_A1 @ self.Wdy
            + M_A0_D_A1 @ self.Wmy
            + self._outcome_square(W, D_A1, M_A0_D_A1, X)
        )
        return float(Y.mean())

    def population_true_psi(self):
        a1 = np.ones((1, 1))
        a0 = np.zeros((1, 1))
        d_shift = a1 @ self.Wad
        m_shift = a0 @ self.Wam + d_shift @ self.Wdm
        y_shift = a1 @ self.Way + d_shift @ self.Wdy + m_shift @ self.Wmy
        return float(y_shift.mean())

    def true_psi_x(self, data):
        U = data["u"]
        X = data["x"]
        W = data["w"]
        n = X.shape[0]

        A1 = np.ones((n, 1))
        A0 = np.zeros((n, 1))
        D_A1 = U @ self.Wud + X @ self.Wxd + A1 @ self.Wad
        M_A0_D_A1 = U @ self.Wum + X @ self.Wxm + A0 @ self.Wam + D_A1 @ self.Wdm
        return (
            U @ self.Wuy
            + X @ self.Wxy
            + W @ self.Wwy
            + A1 @ self.Way
            + D_A1 @ self.Wdy
            + M_A0_D_A1 @ self.Wmy
            + self._outcome_square(W, D_A1, M_A0_D_A1, X)
        )
