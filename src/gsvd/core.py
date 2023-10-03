import numpy as np 
import scipy as sp 

def gsvd(A: np.ndarray, B:np.ndarray, tol: float = 1e-8, return_extras: bool = False):
    """
    Generalized singular value decomposition. 
    """

    if A.shape[1] != B.shape[1]:
        raise ValueError('Last dimension of A & B must be equal.')
    

    p, n = A.shape
    m1, _ = B.shape

    M = np.vstack([A, B])
    m, _ = M.shape
    Q, sv, Zt = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    sv[sv<tol] = 0
    r = np.count_nonzero(sv) # rank(M)
    m2 = m - r

    if p < r:
        raise ValueError('Rows of A must be grather than rank([A, B]).')

    if m1 < n:
        raise ValueError('B matrix must be tall.')
    
    U, CS, Vt = sp.linalg.cossin(Q, p, r, separate=False)

    U_1 = U[:p,:p] # pxp
    U_2 = U[p:,p:] # m1xm1
    V_1t = Vt[:r,:r] # rxr
    V_2t = Vt[r:,r:] # m2xm2

    D_11 = sp.sparse.hstack((CS[:p,:r], sp.sparse.csc_array((p, n-r)))) # pxr + px(n-r)
    D_21 = sp.sparse.hstack((CS[p:,:r], sp.sparse.csc_array((m1, n-r)))) # m1xr + m1x(n-r)

    I = sp.sparse.identity(n-r, format='dia', dtype=np.float64)
    Sr = sp.sparse.dia_matrix((sv[:r], 0), shape=(r,r), dtype=np.float64)
    W = sp.sparse.block_diag((V_1t @ Sr, I)).tocsc()
    X = Zt.T @ sp.sparse.linalg.inv(W)
    
    if return_extras:
        C = CS[:r,:r]
        S = CS[-r:,:r]
        extras = dict( cs=(C, S), rank=r)
        return (U_1, U_2), (D_11, D_21), X, extras
    
    return (U_1, U_2), (D_11, D_21), X