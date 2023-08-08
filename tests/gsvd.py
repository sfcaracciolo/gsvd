from src.gsvd import gsvd
import numpy as np 

rng = np.random.default_rng()

for _ in range(1000):

    m = rng.integers(50, high=100)
    n = rng.integers(50, high=100)

    A = np.random.rand(m, n)
    B = np.random.rand(n, n)

    (U_1, U_2), (D_A, D_B), X = gsvd(A, B)

    assert np.allclose((D_A.T @ D_A + D_B.T @ D_B).todense(), np.eye(n), rtol=1e-5, atol=1e-8)
    assert np.allclose(D_A.todense(), U_1.T @ A @ X, rtol=1e-5, atol=1e-8)
    assert np.allclose(D_B.todense(), U_2.T @ B @ X, rtol=1e-5, atol=1e-8)
    assert np.allclose(U_1 @ D_A @ np.linalg.inv(X), A, rtol=1e-5, atol=1e-8)
    assert np.allclose(U_2 @ D_B @ np.linalg.inv(X), B, rtol=1e-5, atol=1e-8)

print('OK')