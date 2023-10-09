import numpy as np 
import scipy as sp 
from cossin_wrapper import cossin

def gsvd(A: np.ndarray, B:np.ndarray, tol: float = 1e-8):
    """
    Generalized singular value decomposition. 
    """

    if A.shape[1] != B.shape[1]:
        raise ValueError('Last dimension of A & B must be equal.')
    
    if (0 in A.shape) or (0 in B.shape):
        raise ValueError('A and/or B has a dimension of size zero.')

    p, n = A.shape
    m1, n = B.shape

    M = np.vstack([A, B], dtype=np.float64)
    m, n = M.shape

    Q, sv, Zt = sp.linalg.svd(M, full_matrices=True, compute_uv=True, lapack_driver='gesvd')

    sv[sv<tol] = 0
    r = np.count_nonzero(sv) # rank(M)
    m2 = m - r

    if m <= r:
        raise ValueError('Rows of [A,B] must be grather than rank([A, B]).')

    (U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t) = cossin(Q, shape=(p, r), ret='blocks')

    #  (AB)^-1 = B^-1 A^-1
    Srm1 = sp.sparse.dia_matrix((1./sv[:r], 0), shape=(r,r), dtype=np.float64)
    I = sp.sparse.identity(n-r, format='dia', dtype=np.float64)
    X = Zt.T @ sp.sparse.block_diag((Srm1 @ V_1t.T, I))

    D_A = D_11.todia(copy=True)
    D_A.resize((p,n)) # pxr + px(n-r) (padding zeros)
    D_B = D_21.todia(copy=True)
    D_B.resize((m1,n)) # m1xr + m1x(n-r) (padding zeros)

    return (U_1, U_2), (D_A, D_B), X
