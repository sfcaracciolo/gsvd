# GSVD (Generalized Singular Value Decomposition)

A SciPy implementation of the Generalized Singular Value Decomposition [(GSVD)](https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition) based on cosine-sine decomposition [(CS)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cossin.html).

Let $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times n}$, the GSVD is $A = U_A D_A X^{-1}$ and $B = U_B D_B X^{-1}$, where $U_A$, $U_B$ and $X$ are unitary matrices and $D_A$, $D_B$ are diagonal such as $D_A^T D_A + D_B^T D_B = I$.

## Usage

```python

from gsvd import gsvd
import numpy as np

n = rng.integers(50, high=100)
m = rng.integers(50, high=100)

A = np.random.rand(m, n)
B = np.random.rand(n, n)

(U_1, U_2), (D_A, D_B), X = gsvd(A, B)

```