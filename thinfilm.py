import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Function to build FEM matrices
def build_FE_matrices(x, h, local_mass_p1, dt):
    npoint = len(x)
    nelement = npoint - 1

    nd = np.column_stack((np.arange(nelement), np.arange(1, npoint)))
    edet = x[nd[:, 1]] - x[nd[:, 0]]

    S = lil_matrix((npoint, npoint))
    Sw = lil_matrix((npoint, npoint))
    M = lil_matrix((npoint, npoint))
    Dx = lil_matrix((npoint, npoint))

    mobi = np.zeros(nelement)
    for i in range(2):
        for j in range(2):
            mobi += local_mass_p1[i, j] * h[nd[:, i]] * h[nd[:, j]]

    for k in range(nelement):
        dphi = np.array([-1, 1]) / edet[k]
        sloc = np.outer(dphi, dphi) * edet[k]
        mloc = local_mass_p1 * edet[k]
        cloc = np.outer(np.ones(2),dphi) * edet[k] / 2

        indices = nd[k, :]
        for i in range(2):
            for j in range(2):
                S[indices[i], indices[j]] += sloc[i, j]
                Sw[indices[i], indices[j]] += mobi[k] * sloc[i, j]
                M[indices[i], indices[j]] += mloc[i, j]
                Dx[indices[i], indices[j]] += cloc[i, j]

    Z = lil_matrix((npoint, npoint))
    A = csc_matrix(np.block([[M.toarray(), Sw.toarray()], [-dt * S.toarray(), M.toarray()]]))
    return A, S.tocsc(), M.tocsc(), Dx.tocsc()

# Function to build ALE matrices
def build_ALE_matrix(x, h, M, Dx):
    npoint = len(x)
    ndof = npoint

    P = lil_matrix((ndof, ndof))
    I = lil_matrix((ndof, ndof))
    X = lil_matrix((ndof, ndof))

    dh = spsolve(M, Dx @ h)
    xi = (x - x[0]) / (x[-1] - x[0])

    for i in range(ndof):
        P[i, 0] = (1 - xi[i]) * dh[i]
        P[i, -1] = xi[i] * dh[i]
        X[i, 0] = 1 - xi[i]
        X[i, -1] = xi[i]
    for i in range(1, npoint - 1):
        I[i, i] = 1

    return csc_matrix(P), csc_matrix(I), csc_matrix(X)

# Main function to solve the thin-film equation
def solve_thinfilm():
    # Parameters
    L = 1.0
    T = 0.2
    SL, SR = 1.0, 1.0
    g1, g2 = 0.0, 0.0
    nt = 100
    npoint = 100

    # Initial setup
    x = np.linspace(0, L, npoint)
    h = L / 2 - np.abs(L / 2 - x)

    local_mass_p1 = np.array([[1/3, 1/6], [1/6, 1/3]])
    dt = T / nt

    plt.plot(x, h, 'r--', label='Initial', linewidth=2)

    for it in range(1, nt + 1):
        A, S, M, Dx = build_FE_matrices(x, h, local_mass_p1, dt)
        P, I, X = build_ALE_matrix(x, h, M, Dx)

        rhs = np.zeros(2 * npoint)
        rhs[npoint:] = S @ h + M @ (2 * g2 * h - g1 * (x - x.mean()))

        dh_start = (h[1] - h[0]) / (x[1] - x[0])
        dh_end = (h[-1] - h[-2]) / (x[-1] - x[-2])

        rhs[npoint] += (SL + (dh_start**2) / 2) / abs(dh_start)
        rhs[-1] += (SR + (dh_end**2) / 2) / abs(dh_end)

        u = spsolve(A.tocsc(), rhs)
        U = spsolve(I - P, u[:npoint])
        h += dt * (I @ U)
        x += dt * (X @ U)

        if it % 10 == 1:
            plt.plot(x, h, 'b-', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('h')
    plt.legend()
    plt.savefig("b.png")

if __name__ == "__main__":
    solve_thinfilm()

