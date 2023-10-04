from matplotlib import pyplot as plt
import numpy as np 
import scipy as sp 

def gsvd(A: np.ndarray, B:np.ndarray, tol: float = 1e-8, return_extras: bool = False):
    """
    Generalized singular value decomposition. 
    """

    if A.shape[1] != B.shape[1]:
        raise ValueError('Last dimension of A & B must be equal.')
    

    p, n = A.shape
    m1, n = B.shape

    M = np.vstack([A, B])
    m, n = M.shape

    Q, sv, Zt = np.linalg.svd(M, full_matrices=True, compute_uv=True)

    sv[sv<tol] = 0
    r = np.count_nonzero(sv) # rank(M)
    m2 = m - r

    if m <= r:
        raise ValueError('Rows of [A,B] must be grather than rank([A, B]).')

    # if p < r:
    #     raise ValueError('Rows of A must be grather than rank([A, B]).')

    # if m1 < r:
    #     raise ValueError('Rows of B must be grather than rank([A, B]).')
    
    U, D, Vt = sp.linalg.cossin(Q, p=p, q=r, separate=False)

    U_1 = U[:p,:p] # pxp
    U_2 = U[p:,p:] # m1xm1
    V_1t = Vt[:r,:r] # rxr
    V_2t = Vt[r:,r:] # m2xm2

    D_11, D_12 = D[:p,:r], D[:p,r:]
    D_21, D_22 = D[p:,:r], D[p:,r:]

    I = sp.sparse.identity(n-r, format='dia', dtype=np.float64)
    Sr = sp.sparse.dia_matrix((sv[:r], 0), shape=(r,r), dtype=np.float64)
    W = sp.sparse.block_diag((V_1t @ Sr, I)).tocsc()
    X = Zt.T @ sp.sparse.linalg.inv(W)
    
    w = min([p, r, m1, m2])
    fmin = lambda a, b: min([a, b])-w
    fmax = lambda a, b: max([a-b, 0])

    a = fmin(p,r)
    C = D_11[a:a+w, a:a+w]

    b = fmax(m1,r)
    c = fmax(r,m1)
    print(b, c)
    S = D_21[b:b+w, c:c+w]

    print(np.allclose(Q, U@D@Vt))
    plt.spy(D)
    plt.spy(D_21)
    plt.show()
    D_A = sp.sparse.hstack((D_11, sp.sparse.csc_array(( p, n-r)))) # pxr + px(n-r)
    D_B = sp.sparse.hstack((D_21, sp.sparse.csc_array((m1, n-r)))) # m1xr + m1x(n-r)

    if return_extras: return (U_1, U_2), (D_A, D_B), X, dict(cs=(C, S), rank=r)
    return (U_1, U_2), (D_A, D_B), X