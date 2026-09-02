"""NumPy-only shape test for the review-version joint KKT systems.

This test exists so the derivation can be checked on machines without PyTorch.
The production estimator in this folder uses PyTorch/CUDA.
"""

import numpy as np


def rbf(x, y, gamma=0.2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xx = (x * x).sum(axis=1)[:, None]
    yy = (y * y).sum(axis=1)[None, :]
    return np.exp(-gamma * np.maximum(xx + yy - 2.0 * x @ y.T, 0.0))


def safe_solve(mat, rhs):
    mat = 0.5 * (mat + mat.T)
    try:
        return np.linalg.solve(mat + 1e-8 * np.eye(mat.shape[0]), rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(mat + 1e-6 * np.eye(mat.shape[0]), rhs, rcond=None)[0]


def second_stage_h_system(w2, w1, z2, z1, a, y):
    n = len(a)
    loc1 = a[:, 0] == 1
    loc0 = ~loc1
    kh1 = rbf(w1, w1)
    kh2 = rbf(w2, w2)
    kf1 = rbf(z1, z1)
    kf2 = rbf(z2, z2)
    lam = 1e-2
    c0 = 1.0 / loc0.sum()
    c1 = 1.0 / loc1.sum()

    mat = np.zeros((4 * n, 4 * n))
    rhs = np.zeros(4 * n)
    ah1, ah2, bf1, bf2 = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n), slice(3 * n, 4 * n)

    h10, h20, f10 = kh1[loc0], kh2[loc0], kf1[loc0]
    h21, f21 = kh2[loc1], kf2[loc1]
    s1 = kf1.T @ kf1 / n + lam * kf1
    s2 = kf2.T @ kf2 / n + lam * kf2

    mat[ah1, ah1] = 2 * lam * kh1
    mat[ah1, bf1] = -c0 * h10.T @ f10
    mat[ah2, ah2] = 2 * lam * kh2
    mat[ah2, bf1] = c0 * h20.T @ f10
    mat[ah2, bf2] = -c1 * h21.T @ f21
    mat[bf1, ah1] = mat[ah1, bf1].T
    mat[bf1, ah2] = mat[ah2, bf1].T
    mat[bf1, bf1] = -2 * s1
    mat[bf2, ah2] = mat[ah2, bf2].T
    mat[bf2, bf2] = -2 * s2
    rhs[bf2] = -c1 * f21.T @ y[loc1, 0]
    return safe_solve(mat, rhs)[ah1]


def third_stage_q_system(q2, q1, q0, h2, h1, h0, a):
    n = len(a)
    aa = a[:, 0]
    one = np.ones(n)
    kq2, kq1, kq0 = rbf(q2, q2), rbf(q1, q1), rbf(q0, q0)
    kf2, kf1, kf0 = rbf(h2, h2), rbf(h1, h1), rbf(h0, h0)
    lam = 1e-2
    c = 1.0 / n
    mat = np.zeros((6 * n, 6 * n))
    rhs = np.zeros(6 * n)
    aq2, aq1, aq0 = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
    bf2, bf1, bf0 = slice(3 * n, 4 * n), slice(4 * n, 5 * n), slice(5 * n, 6 * n)

    def wc(left, weights, right):
        return left.T @ (weights[:, None] * right)

    mat[aq2, aq2] = 2 * lam * kq2
    mat[aq2, bf2] = c * wc(kq2, aa, kf2)
    mat[aq1, aq1] = 2 * lam * kq1
    mat[aq1, bf2] = -c * wc(kq1, one - aa, kf2)
    mat[aq1, bf1] = c * wc(kq1, one - aa, kf1)
    mat[aq0, aq0] = 2 * lam * kq0
    mat[aq0, bf1] = -c * wc(kq0, aa, kf1)
    mat[aq0, bf0] = c * wc(kq0, aa, kf0)
    for left, right in [(aq2, bf2), (aq1, bf2), (aq1, bf1), (aq0, bf1), (aq0, bf0)]:
        mat[right, left] = mat[left, right].T
    for slc, gram in [(bf2, kf2), (bf1, kf1), (bf0, kf0)]:
        mat[slc, slc] = -2 * (gram.T @ gram / n + lam * gram)
    rhs[bf0] = c * kf0.T @ one
    return safe_solve(mat, rhs)[aq2]


def main():
    rng = np.random.default_rng(7)
    n = 30
    x = rng.normal(size=(n, 2))
    z = rng.normal(size=(n, 2))
    w = rng.normal(size=(n, 2))
    d = rng.normal(size=(n, 1))
    m = rng.normal(size=(n, 1))
    a = (rng.uniform(size=(n, 1)) < 0.5).astype(float)
    y = x[:, [0]] + w[:, [0]] + d + m + a + 0.1 * rng.normal(size=(n, 1))

    h2 = np.hstack((w, x, m, d))
    h1 = np.hstack((w, d, x))
    h0 = np.hstack((w, x))
    q2 = np.hstack((z, x, m, d))
    q1 = np.hstack((z, d, x))
    q0 = np.hstack((z, x))

    alpha_h1 = second_stage_h_system(h2, h1, q2, q1, a, y)
    alpha_q2 = third_stage_q_system(q2, q1, q0, h2, h1, h0, a)

    assert alpha_h1.shape == (n,)
    assert alpha_q2.shape == (n,)
    assert np.isfinite(alpha_h1).all()
    assert np.isfinite(alpha_q2).all()
    print("numpy joint KKT test ok")


if __name__ == "__main__":
    main()
