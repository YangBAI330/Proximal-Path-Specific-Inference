from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str = "float64") -> torch.dtype:
    if dtype in ("float64", "double"):
        return torch.float64
    if dtype in ("float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def to_tensor(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).T
    dist2 = torch.clamp(x_norm + y_norm - 2.0 * (x @ y.T), min=0.0)
    return torch.exp(-gamma * dist2)


def fit_standardizer(x: torch.Tensor, eps: float = 1e-8):
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / scale, mean, scale


def apply_standardizer(x: torch.Tensor, mean: Optional[torch.Tensor], scale: Optional[torch.Tensor]) -> torch.Tensor:
    if mean is None or scale is None:
        return x
    return (x - mean) / scale


def resolve_gamma(x: torch.Tensor, raw_gamma: float) -> float:
    if raw_gamma > 0:
        return float(raw_gamma)
    with torch.no_grad():
        x_norm = (x * x).sum(dim=1, keepdim=True)
        dist2 = torch.clamp(x_norm + x_norm.T - 2.0 * (x @ x.T), min=0.0)
        mask = ~torch.eye(dist2.shape[0], device=dist2.device, dtype=torch.bool)
        vals = dist2[mask]
        vals = vals[vals > 1e-12]
        if vals.numel() == 0:
            return 1.0
        return float((1.0 / torch.median(vals).clamp_min(1e-6)).item())


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def safe_solve(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    jitter: float = 1e-8,
    max_tries: int = 6,
) -> torch.Tensor:
    matrix = symmetrize(matrix)
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    scale = torch.linalg.norm(matrix).detach().clamp_min(1.0)
    last_error: Optional[Exception] = None

    for i in range(max_tries):
        add = jitter * (10.0**i) * scale
        try:
            sol = torch.linalg.solve(matrix + add * eye, rhs)
            if torch.isfinite(sol).all():
                return sol
        except RuntimeError as exc:
            last_error = exc

    try:
        return torch.linalg.pinv(matrix + jitter * (10.0**max_tries) * scale * eye) @ rhs
    except RuntimeError:
        if last_error is not None:
            raise last_error
        raise


def weighted_cross(left: torch.Tensor, weights: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left.T @ (weights.reshape(-1, 1) * right)


@dataclass
class KernelFunction:
    centers: torch.Tensor
    alpha: torch.Tensor
    gamma: float
    mean: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None

    def __call__(self, x) -> torch.Tensor:
        x_t = to_tensor(x, self.centers.device, self.centers.dtype)
        x_t = apply_standardizer(x_t, self.mean, self.scale)
        return rbf_kernel(x_t, self.centers, self.gamma) @ self.alpha

    def numpy(self, x) -> np.ndarray:
        return self(x).detach().cpu().numpy()
