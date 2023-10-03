# GSVD (Generalized Singular Value Decomposition)

A SciPy implementation of the Generalized Singular Value Decomposition [(GSVD)](https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition) based on cosine-sine decomposition [(CS)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cossin.html).

Let $A \in \mathbb{R}^{p \times n}$ and $B \in \mathbb{R}^{(m-p) \times n}$, $p \ge \text{rank}(\text{vstack}(A, B))$ and $m-p \ge n$, the GSVD is $A = U_1 D_A X^{-1}$ and $B = U_2 D_B X^{-1}$, where $U_1$, $U_2$ and are unitary matrices of shape $p$ and $m-p$. $X$ is nonsingular with shape $n$. $D_A$, $D_B$ are diagonal.

## Usage

```python

from gsvd import gsvd
import numpy as np

p = rng.integers(50, high=100)
n = rng.integers(25, high=50)
m1 = rng.integers(50, high=150)

A = np.random.rand(p, n)
B = np.random.rand(m1, n)

(U_1, U_2), (D_A, D_B), X = gsvd(A, B)

```