import numpy as np 
import scipy as sp 

def gsvd(A: np.ndarray, L:np.ndarray):
    """
    Generalized singular value decomposition.
    """
    p, q = A.shape
    m = p + q
    M = np.vstack([A, L])
    Q, S, Zt = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    U, CS, Vt = sp.linalg.cossin(Q, p, q, separate=False)
    r = S.size # rank(M)

    U_1 = U[:p,:p] # pxp
    U_2 = U[p:,p:] # qxq
    V_1 = Vt.T[:r,:r] # rxr

    I = sp.sparse.identity(q-r, dtype=np.float64, format='dia')
    Sr = sp.sparse.dia_matrix((S, 0), shape=(r,r), dtype=np.float64)
    T = sp.sparse.block_diag((V_1.T @ Sr, I))
    X = Zt.T @ sp.linalg.inv(T.todense())

    D_A = sp.sparse.dia_matrix(CS[:p,:q], dtype=np.float64)
    D_L = sp.sparse.dia_matrix(CS[p:m,:q], dtype=np.float64)
    
    return (U_1, U_2), (D_A, D_L), X