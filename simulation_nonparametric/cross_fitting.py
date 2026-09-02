import numpy as np

try:
    from .all_estimators import AllEstimator
    from .bridge_estimators import BridgeConfig
except ImportError:
    from all_estimators import AllEstimator
    from bridge_estimators import BridgeConfig


Z_975 = 1.959963984540054
OBSERVED_KEYS = ("x", "z", "w", "a", "d", "m", "y")


def _kfold_indices(n, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    for test_idx in folds:
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        yield train_idx, test_idx


def _subset(data, idx):
    return {k: v[idx] for k, v in data.items()}


def _observed_only(data):
    return {k: data[k] for k in OBSERVED_KEYS}


def cross_fitting_estimate(datagen, data_all, config: BridgeConfig = None, n_splits=5, seed=42):
    sample_true_psi = datagen.true_psi(data_all)
    population_true_psi = datagen.population_true_psi()
    n_total = len(data_all["x"])

    sample_values = {name: np.zeros(n_total) for name in ["por", "pipw", "phe1", "phe2", "pmr"]}

    for fold_idx, (train_idx, test_idx) in enumerate(_kfold_indices(n_total, n_splits, seed=seed), 1):
        data_fit = _observed_only(_subset(data_all, train_idx))
        data_test = _observed_only(_subset(data_all, test_idx))

        model = AllEstimator(config=config)
        try:
            model.fit(data_fit)
            por = np.asarray(model.evaluate_por(data_test))
            pipw = np.asarray(model.evaluate_pipw(data_test))
            phe1 = np.asarray(model.evaluate_phe1(data_test))
            phe2 = np.asarray(model.evaluate_phe2(data_test))
            pmr = np.asarray(model.evaluate_pmr(data_test))
        except Exception as exc:
            raise RuntimeError(f"Fold {fold_idx} failed: {exc}") from exc

        sample_values["por"][test_idx] = por
        sample_values["pipw"][test_idx] = pipw
        sample_values["phe1"][test_idx] = phe1
        sample_values["phe2"][test_idx] = phe2
        sample_values["pmr"][test_idx] = pmr

    estimates = {k: float(np.mean(v)) for k, v in sample_values.items()}
    sample_if_pmr = sample_values["pmr"] - estimates["pmr"]
    if_var = float(np.var(sample_if_pmr, ddof=1)) if n_total > 1 else 0.0
    pmr_se = float(np.sqrt(if_var / n_total))
    ci_lower = estimates["pmr"] - Z_975 * pmr_se
    ci_upper = estimates["pmr"] + Z_975 * pmr_se

    estimates.update(
        {
            "pmr_se": pmr_se,
            "pmr_ci_lower": float(ci_lower),
            "pmr_ci_upper": float(ci_upper),
            "pmr_ci_cover": bool(ci_lower <= population_true_psi <= ci_upper),
            "pmr_ci_cover_sample": bool(ci_lower <= sample_true_psi <= ci_upper),
            "pmr_ci_width": float(ci_upper - ci_lower),
            "pmr_if_variance": if_var,
            "sample_true_psi": float(sample_true_psi),
            "population_true_psi": float(population_true_psi),
        }
    )
    return estimates, population_true_psi
