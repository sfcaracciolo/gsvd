from matplotlib import pyplot as plt
from src.gsvd import cs, cs2
import numpy as np 
import scipy  as sp 

rng = np.random.default_rng()

for _ in range(100):

    m = rng.integers(50, high=100)
    d = np.random.randint(1, m-1)
    Q = sp.stats.unitary_group.rvs(m)
    C, S = cs(Q, shape=(d, m-d), ret='cs')
    C2, S2 = cs2(Q, shape=(d, m-d), ret='cs')

    assert np.allclose(C.todense(), C2.todense(), rtol=1e-5, atol=1e-8)
    assert np.allclose(S.todense(), S2.todense(), rtol=1e-5, atol=1e-8)

print('OK')