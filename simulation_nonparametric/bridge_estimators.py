from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from .kernels import (
        KernelFunction,
        fit_standardizer,
        rbf_kernel,
        resolve_device,
        resolve_dtype,
        resolve_gamma,
        safe_solve,
        to_tensor,
        weighted_cross,
    )
except ImportError:
    from kernels import (
        KernelFunction,
        fit_standardizer,
        rbf_kernel,
        resolve_device,
        resolve_dtype,
        resolve_gamma,
        safe_solve,
        to_tensor,
        weighted_cross,
    )


@dataclass
class BridgeConfig:
    lambda_bridge: float = 1e-3
    lambda_adv: float = 1e-2
    lambda_power: float = 1.0
    adaptive_lambda: bool = False
    gamma_h: float = 0.0
    gamma_q: float = 0.0
    gamma_f: float = 0.0
    jitter: float = 1e-7
    device: str = "auto"
    dtype: str = "float64"
    penalty: str = "l2"
    standardize: bool = True
    q_clip: float = 10.0
    h_clip: float = 20.0
    score_clip: float = 10.0


def _concat(*xs: torch.Tensor) -> torch.Tensor:
    return torch.cat(xs, dim=1)


def tensorize_data(data: Dict[str, np.ndarray], config: BridgeConfig) -> Dict[str, torch.Tensor]:
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    return {k: to_tensor(v, device, dtype) for k, v in data.items()}


def feature_sets(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    w, x, z, d, m = data["w"], data["x"], data["z"], data["d"], data["m"]
    return {
        "h2": _concat(w, x, m, d),
        "h1": _concat(w, d, x),
        "h0": _concat(w, x),
        "q2": _concat(z, x, m, d),
        "q1": _concat(z, d, x),
        "q0": _concat(z, x),
    }


def _bridge_penalty(
    gram: torch.Tensor,
    eval_kernel: torch.Tensor,
    config: BridgeConfig,
) -> torch.Tensor:
    lam = _effective_lambda(config.lambda_bridge, gram.shape[0], config)
    if config.penalty == "rkhs":
        return lam * gram
    if config.penalty == "l2":
        n = max(1, eval_kernel.shape[0])
        return lam * (eval_kernel.T @ eval_kernel) / float(n)
    raise ValueError(f"Unsupported penalty: {config.penalty}")


def _effective_lambda(raw_lambda: float, n: int, config: BridgeConfig) -> float:
    if config.adaptive_lambda:
        return raw_lambda / (float(max(1, n)) ** config.lambda_power)
    return raw_lambda


def _adv_square(
    eval_kernel: torch.Tensor,
    gram: torch.Tensor,
    config: BridgeConfig,
) -> torch.Tensor:
    n = max(1, eval_kernel.shape[0])
    lam = _effective_lambda(config.lambda_adv, gram.shape[0], config)
    return (eval_kernel.T @ eval_kernel) / float(n) + lam * gram


def _empty_check(mask: torch.Tensor, name: str) -> None:
    if int(mask.sum().item()) == 0:
        raise ValueError(f"No observations available for {name}")


def _prepare_features(x: torch.Tensor, config: BridgeConfig):
    if config.standardize:
        return fit_standardizer(x)
    return x, None, None


def _gram(x: torch.Tensor, raw_gamma: float) -> Tuple[torch.Tensor, float]:
    gamma = resolve_gamma(x, raw_gamma)
    return rbf_kernel(x, x, gamma), gamma


def _single_minimax(
    bridge_x: torch.Tensor,
    adv_x: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    config: BridgeConfig,
    gamma_bridge: float,
    gamma_adv: float,
    mask: torch.Tensor = None,
) -> KernelFunction:
    n = bridge_x.shape[0]
    if mask is None:
        mask = torch.ones(n, device=bridge_x.device, dtype=torch.bool)
    _empty_check(mask, "single-stage minimax")

    bridge_x, bridge_mean, bridge_scale = _prepare_features(bridge_x, config)
    adv_x, _, _ = _prepare_features(adv_x, config)
    kh, gamma_bridge_resolved = _gram(bridge_x, gamma_bridge)
    kf, _ = _gram(adv_x, gamma_adv)
    hm = kh[mask]
    fm = kf[mask]
    fs = kf
    gm1 = g1[mask].reshape(-1)
    gm2 = g2[mask].reshape(-1)

    c_m = 1.0 / float(mask.sum().item())
    p_h = _bridge_penalty(kh, kh, config)
    s_f = _adv_square(fs, kf, config)

    size = 2 * n
    mat = torch.zeros((size, size), device=bridge_x.device, dtype=bridge_x.dtype)
    rhs = torch.zeros(size, device=bridge_x.device, dtype=bridge_x.dtype)
    a = slice(0, n)
    b = slice(n, 2 * n)

    cross = c_m * weighted_cross(hm, gm1, fm)
    mat[a, a] = 2.0 * p_h
    mat[a, b] = cross
    mat[b, a] = cross.T
    mat[b, b] = -2.0 * s_f
    rhs[b] = -c_m * (fm.T @ gm2)

    sol = safe_solve(mat, rhs, jitter=config.jitter)
    return KernelFunction(
        bridge_x.detach(),
        sol[a].detach(),
        gamma_bridge_resolved,
        None if bridge_mean is None else bridge_mean.detach(),
        None if bridge_scale is None else bridge_scale.detach(),
    )


def solve_h2(data_t: Dict[str, torch.Tensor], config: BridgeConfig) -> KernelFunction:
    feats = feature_sets(data_t)
    a = data_t["a"].reshape(-1)
    y = data_t["y"].reshape(-1)
    mask = a > 0.5
    return _single_minimax(
        feats["h2"],
        feats["q2"],
        -torch.ones_like(a),
        y,
        config,
        config.gamma_h,
        config.gamma_f,
        mask,
    )


def solve_q0(data_t: Dict[str, torch.Tensor], config: BridgeConfig) -> KernelFunction:
    feats = feature_sets(data_t)
    a = data_t["a"].reshape(-1)
    return _single_minimax(
        feats["q0"],
        feats["h0"],
        a,
        -torch.ones_like(a),
        config,
        config.gamma_q,
        config.gamma_f,
    )


def _slices(blocks):
    out = []
    start = 0
    for width in blocks:
        out.append(slice(start, start + width))
        start += width
    return out


def solve_h1_joint(data_t: Dict[str, torch.Tensor], config: BridgeConfig) -> Tuple[KernelFunction, KernelFunction]:
    feats = feature_sets(data_t)
    h1_x, h2_x = feats["h1"], feats["h2"]
    f1_x, f2_x = feats["q1"], feats["q2"]
    a = data_t["a"].reshape(-1)
    y = data_t["y"].reshape(-1)
    mask0 = a < 0.5
    mask1 = a > 0.5
    _empty_check(mask0, "h1 A=0 moment")
    _empty_check(mask1, "h1 A=1 moment")

    n = a.shape[0]
    h1_x, h1_mean, h1_scale = _prepare_features(h1_x, config)
    h2_x, h2_mean, h2_scale = _prepare_features(h2_x, config)
    f1_x, _, _ = _prepare_features(f1_x, config)
    f2_x, _, _ = _prepare_features(f2_x, config)
    kh1, gamma_h1 = _gram(h1_x, config.gamma_h)
    kh2, gamma_h2 = _gram(h2_x, config.gamma_h)
    kf1, _ = _gram(f1_x, config.gamma_f)
    kf2, _ = _gram(f2_x, config.gamma_f)

    p_h1 = _bridge_penalty(kh1, kh1, config)
    p_h2 = _bridge_penalty(kh2, kh2, config)
    s_f1 = _adv_square(kf1, kf1, config)
    s_f2 = _adv_square(kf2, kf2, config)
    c0 = 1.0 / float(mask0.sum().item())
    c1 = 1.0 / float(mask1.sum().item())

    h10, h20, f10 = kh1[mask0], kh2[mask0], kf1[mask0]
    h21, f21 = kh2[mask1], kf2[mask1]

    mat = torch.zeros((4 * n, 4 * n), device=a.device, dtype=a.dtype)
    rhs = torch.zeros(4 * n, device=a.device, dtype=a.dtype)
    ah1, ah2, bf1, bf2 = _slices([n, n, n, n])

    mat[ah1, ah1] = 2.0 * p_h1
    mat[ah1, bf1] = -c0 * (h10.T @ f10)

    mat[ah2, ah2] = 2.0 * p_h2
    mat[ah2, bf1] = c0 * (h20.T @ f10)
    mat[ah2, bf2] = -c1 * (h21.T @ f21)

    mat[bf1, ah1] = mat[ah1, bf1].T
    mat[bf1, ah2] = mat[ah2, bf1].T
    mat[bf1, bf1] = -2.0 * s_f1

    mat[bf2, ah2] = mat[ah2, bf2].T
    mat[bf2, bf2] = -2.0 * s_f2
    rhs[bf2] = -c1 * (f21.T @ y[mask1])

    sol = safe_solve(mat, rhs, jitter=config.jitter)
    h1 = KernelFunction(h1_x.detach(), sol[ah1].detach(), gamma_h1, h1_mean, h1_scale)
    h2_aux = KernelFunction(h2_x.detach(), sol[ah2].detach(), gamma_h2, h2_mean, h2_scale)
    return h1, h2_aux


def solve_h0_joint(data_t: Dict[str, torch.Tensor], config: BridgeConfig) -> Tuple[KernelFunction, KernelFunction, KernelFunction]:
    feats = feature_sets(data_t)
    h0_x, h1_x, h2_x = feats["h0"], feats["h1"], feats["h2"]
    f0_x, f1_x, f2_x = feats["q0"], feats["q1"], feats["q2"]
    a = data_t["a"].reshape(-1)
    y = data_t["y"].reshape(-1)
    mask0 = a < 0.5
    mask1 = a > 0.5
    _empty_check(mask0, "h0 A=0 moment")
    _empty_check(mask1, "h0 A=1 moment")

    n = a.shape[0]
    h0_x, h0_mean, h0_scale = _prepare_features(h0_x, config)
    h1_x, h1_mean, h1_scale = _prepare_features(h1_x, config)
    h2_x, h2_mean, h2_scale = _prepare_features(h2_x, config)
    f0_x, _, _ = _prepare_features(f0_x, config)
    f1_x, _, _ = _prepare_features(f1_x, config)
    f2_x, _, _ = _prepare_features(f2_x, config)
    kh0, gamma_h0 = _gram(h0_x, config.gamma_h)
    kh1, gamma_h1 = _gram(h1_x, config.gamma_h)
    kh2, gamma_h2 = _gram(h2_x, config.gamma_h)
    kf0, _ = _gram(f0_x, config.gamma_f)
    kf1, _ = _gram(f1_x, config.gamma_f)
    kf2, _ = _gram(f2_x, config.gamma_f)

    p_h0 = _bridge_penalty(kh0, kh0, config)
    p_h1 = _bridge_penalty(kh1, kh1, config)
    p_h2 = _bridge_penalty(kh2, kh2, config)
    s_f0 = _adv_square(kf0, kf0, config)
    s_f1 = _adv_square(kf1, kf1, config)
    s_f2 = _adv_square(kf2, kf2, config)
    c0 = 1.0 / float(mask0.sum().item())
    c1 = 1.0 / float(mask1.sum().item())

    h01, h11, f01 = kh0[mask1], kh1[mask1], kf0[mask1]
    h10, h20, f10 = kh1[mask0], kh2[mask0], kf1[mask0]
    h21, f21 = kh2[mask1], kf2[mask1]

    mat = torch.zeros((6 * n, 6 * n), device=a.device, dtype=a.dtype)
    rhs = torch.zeros(6 * n, device=a.device, dtype=a.dtype)
    ah0, ah1, ah2, bf0, bf1, bf2 = _slices([n, n, n, n, n, n])

    mat[ah0, ah0] = 2.0 * p_h0
    mat[ah0, bf0] = -c1 * (h01.T @ f01)

    mat[ah1, ah1] = 2.0 * p_h1
    mat[ah1, bf0] = c1 * (h11.T @ f01)
    mat[ah1, bf1] = -c0 * (h10.T @ f10)

    mat[ah2, ah2] = 2.0 * p_h2
    mat[ah2, bf1] = c0 * (h20.T @ f10)
    mat[ah2, bf2] = -c1 * (h21.T @ f21)

    for left, right in [(ah0, bf0), (ah1, bf0), (ah1, bf1), (ah2, bf1), (ah2, bf2)]:
        mat[right, left] = mat[left, right].T
    mat[bf0, bf0] = -2.0 * s_f0
    mat[bf1, bf1] = -2.0 * s_f1
    mat[bf2, bf2] = -2.0 * s_f2
    rhs[bf2] = -c1 * (f21.T @ y[mask1])

    sol = safe_solve(mat, rhs, jitter=config.jitter)
    h0 = KernelFunction(h0_x.detach(), sol[ah0].detach(), gamma_h0, h0_mean, h0_scale)
    h1_aux = KernelFunction(h1_x.detach(), sol[ah1].detach(), gamma_h1, h1_mean, h1_scale)
    h2_aux = KernelFunction(h2_x.detach(), sol[ah2].detach(), gamma_h2, h2_mean, h2_scale)
    return h0, h1_aux, h2_aux


def solve_q1_joint(data_t: Dict[str, torch.Tensor], config: BridgeConfig) -> Tuple[KernelFunction, KernelFunction]:
    feats = feature_sets(data_t)
    q1_x, q0_x = feats["q1"], feats["q0"]
    f1_x, f0_x = feats["h1"], feats["h0"]
    a = data_t["a"].reshape(-1)
    one = torch.ones_like(a)
    n = a.shape[0]
    c = 1.0 / float(n)

    q1_x, q1_mean, q1_scale = _prepare_features(q1_x, config)
    q0_x, q0_mean, q0_scale = _prepare_features(q0_x, config)
    f1_x, _, _ = _prepare_features(f1_x, config)
    f0_x, _, _ = _prepare_features(f0_x, config)
    kq1, gamma_q1 = _gram(q1_x, config.gamma_q)
    kq0, gamma_q0 = _gram(q0_x, config.gamma_q)
    kf1, _ = _gram(f1_x, config.gamma_f)
    kf0, _ = _gram(f0_x, config.gamma_f)

    p_q1 = _bridge_penalty(kq1, kq1, config)
    p_q0 = _bridge_penalty(kq0, kq0, config)
    s_f1 = _adv_square(kf1, kf1, config)
    s_f0 = _adv_square(kf0, kf0, config)

    mat = torch.zeros((4 * n, 4 * n), device=a.device, dtype=a.dtype)
    rhs = torch.zeros(4 * n, device=a.device, dtype=a.dtype)
    aq1, aq0, bf1, bf0 = _slices([n, n, n, n])

    mat[aq1, aq1] = 2.0 * p_q1
    mat[aq1, bf1] = c * weighted_cross(kq1, one - a, kf1)

    mat[aq0, aq0] = 2.0 * p_q0
    mat[aq0, bf1] = -c * weighted_cross(kq0, a, kf1)
    mat[aq0, bf0] = c * weighted_cross(kq0, a, kf0)

    mat[bf1, aq1] = mat[aq1, bf1].T
    mat[bf1, aq0] = mat[aq0, bf1].T
    mat[bf1, bf1] = -2.0 * s_f1

    mat[bf0, aq0] = mat[aq0, bf0].T
    mat[bf0, bf0] = -2.0 * s_f0
    rhs[bf0] = c * (kf0.T @ one)

    sol = safe_solve(mat, rhs, jitter=config.jitter)
    q1 = KernelFunction(q1_x.detach(), sol[aq1].detach(), gamma_q1, q1_mean, q1_scale)
    q0_aux = KernelFunction(q0_x.detach(), sol[aq0].detach(), gamma_q0, q0_mean, q0_scale)
    return q1, q0_aux


def solve_q2_joint(data_t: Dict[str, torch.Tensor], config: BridgeConfig) -> Tuple[KernelFunction, KernelFunction, KernelFunction]:
    feats = feature_sets(data_t)
    q2_x, q1_x, q0_x = feats["q2"], feats["q1"], feats["q0"]
    f2_x, f1_x, f0_x = feats["h2"], feats["h1"], feats["h0"]
    a = data_t["a"].reshape(-1)
    one = torch.ones_like(a)
    n = a.shape[0]
    c = 1.0 / float(n)

    q2_x, q2_mean, q2_scale = _prepare_features(q2_x, config)
    q1_x, q1_mean, q1_scale = _prepare_features(q1_x, config)
    q0_x, q0_mean, q0_scale = _prepare_features(q0_x, config)
    f2_x, _, _ = _prepare_features(f2_x, config)
    f1_x, _, _ = _prepare_features(f1_x, config)
    f0_x, _, _ = _prepare_features(f0_x, config)
    kq2, gamma_q2 = _gram(q2_x, config.gamma_q)
    kq1, gamma_q1 = _gram(q1_x, config.gamma_q)
    kq0, gamma_q0 = _gram(q0_x, config.gamma_q)
    kf2, _ = _gram(f2_x, config.gamma_f)
    kf1, _ = _gram(f1_x, config.gamma_f)
    kf0, _ = _gram(f0_x, config.gamma_f)

    p_q2 = _bridge_penalty(kq2, kq2, config)
    p_q1 = _bridge_penalty(kq1, kq1, config)
    p_q0 = _bridge_penalty(kq0, kq0, config)
    s_f2 = _adv_square(kf2, kf2, config)
    s_f1 = _adv_square(kf1, kf1, config)
    s_f0 = _adv_square(kf0, kf0, config)

    mat = torch.zeros((6 * n, 6 * n), device=a.device, dtype=a.dtype)
    rhs = torch.zeros(6 * n, device=a.device, dtype=a.dtype)
    aq2, aq1, aq0, bf2, bf1, bf0 = _slices([n, n, n, n, n, n])

    mat[aq2, aq2] = 2.0 * p_q2
    mat[aq2, bf2] = c * weighted_cross(kq2, a, kf2)

    mat[aq1, aq1] = 2.0 * p_q1
    mat[aq1, bf2] = -c * weighted_cross(kq1, one - a, kf2)
    mat[aq1, bf1] = c * weighted_cross(kq1, one - a, kf1)

    mat[aq0, aq0] = 2.0 * p_q0
    mat[aq0, bf1] = -c * weighted_cross(kq0, a, kf1)
    mat[aq0, bf0] = c * weighted_cross(kq0, a, kf0)

    for left, right in [(aq2, bf2), (aq1, bf2), (aq1, bf1), (aq0, bf1), (aq0, bf0)]:
        mat[right, left] = mat[left, right].T
    mat[bf2, bf2] = -2.0 * s_f2
    mat[bf1, bf1] = -2.0 * s_f1
    mat[bf0, bf0] = -2.0 * s_f0
    rhs[bf0] = c * (kf0.T @ one)

    sol = safe_solve(mat, rhs, jitter=config.jitter)
    q2 = KernelFunction(q2_x.detach(), sol[aq2].detach(), gamma_q2, q2_mean, q2_scale)
    q1_aux = KernelFunction(q1_x.detach(), sol[aq1].detach(), gamma_q1, q1_mean, q1_scale)
    q0_aux = KernelFunction(q0_x.detach(), sol[aq0].detach(), gamma_q0, q0_mean, q0_scale)
    return q2, q1_aux, q0_aux


class ReviewBridgeSet:
    """Fit all six bridge functions with the review-version joint stages."""

    def __init__(self, config: BridgeConfig = None):
        self.config = config or BridgeConfig()
        self.h2 = None
        self.h1 = None
        self.h0 = None
        self.q0 = None
        self.q1 = None
        self.q2 = None
        self.auxiliary = {}

    def fit(self, data: Dict[str, np.ndarray]):
        data_t = tensorize_data(data, self.config)

        self.h2 = solve_h2(data_t, self.config)
        self.q0 = solve_q0(data_t, self.config)

        self.h1, h2_aux = solve_h1_joint(data_t, self.config)
        self.q1, q0_aux = solve_q1_joint(data_t, self.config)

        self.h0, h1_aux, h2_aux_3 = solve_h0_joint(data_t, self.config)
        self.q2, q1_aux, q0_aux_3 = solve_q2_joint(data_t, self.config)

        self.auxiliary = {
            "h2_from_h1_stage": h2_aux,
            "q0_from_q1_stage": q0_aux,
            "h1_from_h0_stage": h1_aux,
            "h2_from_h0_stage": h2_aux_3,
            "q1_from_q2_stage": q1_aux,
            "q0_from_q2_stage": q0_aux_3,
        }
        return self
