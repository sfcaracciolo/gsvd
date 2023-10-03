from src.gsvd import gsvd
import numpy as np 

rng = np.random.default_rng()

for _ in range(100):

    p = rng.integers(50, high=100)
    n = rng.integers(25, high=50)
    m1 = rng.integers(50, high=150)

    print(p, n, m1)

    A = np.random.rand(p, n)
    B = np.random.rand(m1, n)

    ker = rng.integers(0,n//3, 1)[0]
    A[:,:ker+1] = 1
    B[:,:ker+1] = 1

    (U_1, U_2), (D_A, D_B), X, extras = gsvd(A, B, return_extras=True)

    C, S = extras['cs']
    r = extras['rank']
    
    assert np.allclose((C.T @ C + S.T @ S), np.eye(r), rtol=1e-5, atol=1e-8)
    assert np.allclose(D_A.todense(), U_1.T @ A @ X, rtol=1e-5, atol=1e-8)
    assert np.allclose(D_B.todense(), U_2.T @ B @ X, rtol=1e-5, atol=1e-8)
    assert np.allclose(U_1 @ D_A @ np.linalg.inv(X), A, rtol=1e-5, atol=1e-8)
    assert np.allclose(U_2 @ D_B @ np.linalg.inv(X), B, rtol=1e-5, atol=1e-8)

print('OK')