from matplotlib import pyplot as plt
from src.gsvd import cs
import numpy as np 
import scipy  as sp 

rng = np.random.default_rng()

for _ in range(100):

    m = rng.integers(50, high=100)
    d = np.random.randint(1, m-1)
    Q = sp.stats.unitary_group.rvs(m)
    U, D, Vt = cs(Q, shape=(d, m-d), ret='full')

    assert np.allclose(Q, (U @ D @ Vt).todense(), rtol=1e-5, atol=1e-8)

print('OK')