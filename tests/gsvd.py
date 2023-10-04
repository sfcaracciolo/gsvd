from matplotlib import pyplot as plt
from src.gsvd import gsvd
import numpy as np 

rng = np.random.default_rng()

for _ in range(1000):

    p = rng.integers(25, high=100)
    n = rng.integers(25, high=100)
    m = rng.integers(p+1, high=p+50)

    A = np.random.rand(p, n)
    B = np.random.rand(m-p, n)

    ker = rng.integers(0,n//3, 1)[0]
    A[:,:ker+1] = 1
    B[:,:ker+1] = 1

    r = np.linalg.matrix_rank(np.vstack([A,B]), tol=1e-8)
    print(f'p={p}, r={r}, m1={m-p}, m2={m-r}, n={n}')
    if m <= r:
        continue

    (U_1, U_2), (D_A, D_B), X, extras = gsvd(A, B, return_extras=True)

    C, S = extras['cs']
    w = C.shape[0]
    U, D, Vt = extras['full_cs']
    Q = extras['q']

    # plt.spy(D_B)
    # plt.show()
    assert np.allclose((C.power(2) + S.power(2)).todense(), np.eye(w), rtol=1e-5, atol=1e-8)
    assert np.allclose(D_A.todense(), U_1.T @ A @ X, rtol=1e-5, atol=1e-8)
    assert np.allclose(D_B.todense(), U_2.T @ B @ X, rtol=1e-5, atol=1e-8)
    assert np.allclose(Q, (U @ D @ Vt).todense(), rtol=1e-5, atol=1e-8)
print('OK')