import numpy as np

try:
    from .bridge_estimators import BridgeConfig, ReviewBridgeSet
except ImportError:
    from bridge_estimators import BridgeConfig, ReviewBridgeSet


def _h0_x(data):
    return np.hstack((data["w"], data["x"]))


def _h1_x(data):
    return np.hstack((data["w"], data["d"], data["x"]))


def _h2_x(data):
    return np.hstack((data["w"], data["x"], data["m"], data["d"]))


def _q0_x(data):
    return np.hstack((data["z"], data["x"]))


def _q1_x(data):
    return np.hstack((data["z"], data["d"], data["x"]))


def _q2_x(data):
    return np.hstack((data["z"], data["x"], data["m"], data["d"]))


def _clip(values, limit):
    if limit is None or limit <= 0:
        return values
    return np.clip(values, -float(limit), float(limit))


def _pred(fn, x, clip=None):
    values = fn(x).detach().cpu().numpy().reshape(-1)
    return _clip(values, clip)


class AllEstimator:
    """P-OR, P-IPW, P-hybrid, and P-DML estimators with review bridge fitting."""

    def __init__(self, config: BridgeConfig = None, **config_kwargs):
        if config is not None and config_kwargs:
            raise ValueError("Pass either config or config_kwargs, not both")
        self.config = config or BridgeConfig(**config_kwargs)
        self.bridges = ReviewBridgeSet(self.config)

    @property
    def h2_fn(self):
        return self.bridges.h2

    @property
    def h1_fn(self):
        return self.bridges.h1

    @property
    def h0_fn(self):
        return self.bridges.h0

    @property
    def q0_fn(self):
        return self.bridges.q0

    @property
    def q1_fn(self):
        return self.bridges.q1

    @property
    def q2_fn(self):
        return self.bridges.q2

    def fit(self, fit_data):
        self.bridges.fit(fit_data)
        return self

    def predict_components(self, data):
        return {
            "h0": _pred(self.h0_fn, _h0_x(data), self.config.h_clip),
            "h1": _pred(self.h1_fn, _h1_x(data), self.config.h_clip),
            "h2": _pred(self.h2_fn, _h2_x(data), self.config.h_clip),
            "q0": _pred(self.q0_fn, _q0_x(data), self.config.q_clip),
            "q1": _pred(self.q1_fn, _q1_x(data), self.config.q_clip),
            "q2": _pred(self.q2_fn, _q2_x(data), self.config.q_clip),
        }

    def evaluate_por(self, eval_data):
        return _pred(self.h0_fn, _h0_x(eval_data), self.config.h_clip)

    def evaluate_pipw(self, eval_data):
        a = eval_data["a"][:, 0].astype(float)
        y = eval_data["y"][:, 0].astype(float)
        q2 = _pred(self.q2_fn, _q2_x(eval_data), self.config.q_clip)
        return a * y * q2

    def evaluate_phe1(self, eval_data):
        a = eval_data["a"][:, 0].astype(float)
        h1 = _pred(self.h1_fn, _h1_x(eval_data), self.config.h_clip)
        q0 = _pred(self.q0_fn, _q0_x(eval_data), self.config.q_clip)
        return a * h1 * q0

    def evaluate_phe2(self, eval_data):
        a0 = 1.0 - eval_data["a"][:, 0].astype(float)
        h2 = _pred(self.h2_fn, _h2_x(eval_data), self.config.h_clip)
        q1 = _pred(self.q1_fn, _q1_x(eval_data), self.config.q_clip)
        return a0 * h2 * q1

    def evaluate_pmr(self, eval_data):
        a = eval_data["a"][:, 0].astype(float)
        a0 = 1.0 - a
        y = eval_data["y"][:, 0].astype(float)
        comp = self.predict_components(eval_data)

        term1 = a * comp["q0"] * (comp["h1"] - comp["h0"])
        term2 = a0 * comp["q1"] * (comp["h2"] - comp["h1"])
        term3 = a * comp["q2"] * (y - comp["h2"])
        return _clip(term1 + term2 + term3 + comp["h0"], self.config.score_clip)

    def influence_pmr(self, eval_data, psi_hat):
        return self.evaluate_pmr(eval_data) - psi_hat
