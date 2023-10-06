from typing import Literal, Tuple
from matplotlib import pyplot as plt
import numpy as np 
import scipy as sp 


def cs2(Q: np.ndarray, shape: Tuple[int, int], ret: Literal['full', 'blocks', 'cs'] = 'full'):
    """
    Note: I think cossin function has a issue when separate=False, because in some cases the struct of D is incorrect. 
    The workaround is use separate=True and build each matrix manually.
    FAIL WHEN n < m + p + 1
    """

    p, r = shape
    m, n = Q.shape
    m1, m2 = m - p, m - r 

    U, D, Vt = sp.linalg.cossin(Q, p=p, q=r, separate=False)

    assert np.allclose(U@D@Vt, Q)

    w = min([p, r, m1, m2])
    fmin = lambda a, b: min([a, b])-w
    fmax = lambda a, b: max([a-b, 0])

    D_11 = D[:p, :r]
    D_21 = D[p:, :r]
    D_12 = D[:p, r:]
    D_22 = D[p:, r:]

    U_1, U_2 = U[:p, :p], U[p:, p:]
    V_1t, V_2t = Vt[:r, :r], Vt[r:, r:]

    C = sp.sparse.dia_matrix((D_11[fmin(p,r):fmin(p,r)+w, fmin(p,r):fmin(p,r)+w].diagonal(), 0), shape=(w,w), dtype=np.float64) 
    S = sp.sparse.dia_matrix((D_21[fmax(m1,r):fmax(m1,r)+w, fmax(r,m1):fmax(r,m1)+w].diagonal(), 0), shape=(w,w), dtype=np.float64) 

    if ret == 'full':
        return U, D, Vt

    if ret == 'blocks':
        D_11 = sp.sparse.dia_matrix((D_11.diagonal(), 0), shape=(p,r), dtype=np.float64)
        D_21 = sp.sparse.dia_matrix((D_21.diagonal(), 0), shape=(m1,r), dtype=np.float64)
        D_12 = sp.sparse.dia_matrix((D_12.diagonal(), 0), shape=(p,m2), dtype=np.float64)
        D_22 = sp.sparse.dia_matrix((D_22.diagonal(), 0), shape=(m1,m2), dtype=np.float64)
        return (U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t)

    if ret == 'cs':
        return C, S 
    

def cs(Q: np.ndarray, shape: Tuple[int, int], ret: Literal['full', 'blocks', 'cs'] = 'full'):
    """
    Note: I think cossin function has a issue when separate=False, because in some cases the struct of D is incorrect. 
    The workaround is use separate=True and build each matrix manually.
    """

    p, r = shape
    m, n = Q.shape
    m1, m2 = m - p, m - r 

    (U_1, U_2), theta, (V_1t, V_2t) = sp.linalg.cossin(Q, p=p, q=r, separate=True)

    w = min([p, r, m1, m2])
    fmin = lambda a, b: min([a, b])-w
    fmax = lambda a, b: max([a-b, 0])

    C = sp.sparse.dia_matrix((np.cos(theta), 0), shape=(w,w), dtype=np.float64) 
    S = sp.sparse.dia_matrix((np.sin(theta), 0), shape=(w,w), dtype=np.float64) 

    D_11 = sp.sparse.block_diag((
        sp.sparse.identity(fmin(p,r), format='dia', dtype=np.float64),
        C,
        sp.sparse.csc_array((fmax(p,r), fmax(r,p)), dtype=np.float64)
    ))

    D_21 = sp.sparse.block_diag((
        sp.sparse.csc_array((fmax(m1,r), fmax(r,m1)), dtype=np.float64),
        S,
        sp.sparse.identity(fmin(m1,r), format='dia', dtype=np.float64)
    ))

    D_12 = sp.sparse.block_diag((
        sp.sparse.csc_array((fmax(p,m2), fmax(m2,p)), dtype=np.float64),
        -S,
        -sp.sparse.identity(fmin(m2,p), format='dia', dtype=np.float64)
    ))

    D_22 = sp.sparse.block_diag((
        sp.sparse.identity(fmin(m1, m2), format='dia', dtype=np.float64),
        C,
        sp.sparse.csc_array((fmax(m1,m2), fmax(m2,m1)), dtype=np.float64)
    ))

    if ret == 'full':
        U = sp.sparse.block_diag((U_1, U_2))
        D = sp.sparse.vstack((
            sp.sparse.hstack((D_11, D_12)),
            sp.sparse.hstack((D_21, D_22)),
        ))
        Vt = sp.sparse.block_diag((V_1t, V_2t))
        return U, D, Vt

    if ret == 'blocks':
        return (U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t)

    if ret == 'cs':
        return C, S 
    
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

    (U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t) = cs(Q, shape=(p, r), ret='blocks')

    #  (AB)^-1 = B^-1 A^-1
    Srm1 = sp.sparse.dia_matrix((1./sv[:r], 0), shape=(r,r), dtype=np.float64)
    I = sp.sparse.identity(n-r, format='dia', dtype=np.float64)
    X = Zt.T @ sp.sparse.block_diag((Srm1 @ V_1t.T, I))

    D_A = D_11.todia(copy=True)
    D_A.resize((p,n)) # pxr + px(n-r) (padding zeros)
    D_B = D_21.todia(copy=True)
    D_B.resize((m1,n)) # m1xr + m1x(n-r) (padding zeros)

    return (U_1, U_2), (D_A, D_B), X

# def tkh_factors(A: np.ndarray, B:np.ndarray, tol: float = 1e-8):
#     """
#    Tikhonov factors alpha & beta
#     """

#     if A.shape[1] != B.shape[1]:
#         raise ValueError('Last dimension of A & B must be equal.')
    
#     if (0 in A.shape) or (0 in B.shape):
#         raise ValueError('A and/or B has a dimension of size zero.')

#     p, n = A.shape
#     m1, n = B.shape

#     M = np.vstack([A, B], dtype=np.float64)
#     m, n = M.shape

#     Q, sv, Zt = sp.linalg.svd(M, full_matrices=True, compute_uv=True, lapack_driver='gesvd')

#     sv[sv<tol] = 0
#     r = np.count_nonzero(sv) # rank(M)
#     m2 = m - r

#     if m <= r:
#         raise ValueError('Rows of [A,B] must be grather than rank([A, B]).')

#     C, S = cs(Q, shape=(p, r), ret='cs')

#     w = min([p, r, m1, m2])
#     fmin = lambda a, b: min([a, b])-w
#     fmax = lambda a, b: max([a-b, 0])

#     alphas = np.concatenate((np.ones(fmin(p, r)), C.diagonal(), np.zeros(fmax(r, p))))
#     betas = np.concatenate((np.zeros(fmax(r, m1)), S.diagonal(), np.zeros(fmin(m1, r))))
#     return alphas, betas
    