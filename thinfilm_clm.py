import numpy as np
from scipy.sparse import lil_matrix, diags, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def build_fe_matrices(x, h, nd):
    """
    Builds finite element matrices (S, M, Sw, Dx) for the FEM system.
    """
    nelement = len(nd)
    ndof = len(x)
    edet = x[nd[:, 1]] - x[nd[:, 0]]  # Element lengths

    local_mass_p1 = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])

    S = lil_matrix((ndof, ndof))
    M = lil_matrix((ndof, ndof))
    Sw = lil_matrix((ndof, ndof))
    Dx = lil_matrix((ndof, ndof))

    for k in range(nelement):
        dphi = np.array([-1, 1]) / edet[k]

        sloc = np.outer(dphi, dphi) * edet[k]  # Stiffness matrix
        mloc = local_mass_p1 * edet[k]  # Mass matrix
        cloc = np.outer([1, 1],dphi) / 2 * edet[k]  # Derivative matrix

        h_local = h[nd[k]]
        mobi = np.sum(local_mass_p1 * np.outer(h_local, h_local))

        for i in range(2):
            for j in range(2):
                S[nd[k, i], nd[k, j]] += sloc[i, j]
                M[nd[k, i], nd[k, j]] += mloc[i, j]
                Sw[nd[k, i], nd[k, j]] += mobi * sloc[i, j]
                Dx[nd[k, i], nd[k, j]] += cloc[i, j]

    return S.tocsr(), M.tocsr(), Sw.tocsr(), Dx.tocsr()

def build_ale_matrix(x, h, M, Dx):
    """
    Builds ALE matrices (P, I, X) for ALE decomposition.
    """
    ndof = len(x)
    dh = spsolve(M, Dx @ h)

    xi = (x - x[0]) / (x[-1] - x[0])

    P = lil_matrix((ndof, ndof))
    X = lil_matrix((ndof, ndof))
    I = diags([1] * ndof)

    for i in range(ndof):
        P[i, 0] = (1 - xi[i]) * dh[i]
        P[i, -1] = xi[i] * dh[i]
        X[i, 0] = 1 - xi[i]
        X[i, -1] = xi[i]

    return P.tocsr(), I, X.tocsr()

def thin_film_simulation(L=1.0, T=4.0, SL=1.0, SR=1.0, g1=20.0, g2=0.0, nt=10000, npoint=500, beta=3/4):
    """
    Solves the thin-film equation using FEM and ALE decomposition.
    """
    x = np.linspace(0, L, npoint)
    h = x * (L - x)

    nelement = npoint - 1
    nd = np.array([[i, i + 1] for i in range(nelement)])

    dt = T / nt
    t = 0

    plt.figure()

    for it in range(11):
        S, M, Sw, Dx = build_fe_matrices(x, h, nd)
        P, I, X = build_ale_matrix(x, h, M, Dx)

        DD = beta * np.eye(2)

        # Assemble the global system matrix D
        ndof = len(x)
        D = lil_matrix((ndof + 4, ndof + 4))

        # Core FEM matrices
        D[:ndof, :ndof] = dt * S + M

        # Left boundary constraint
        D[ndof, 0] = 1
        D[ndof + 2, 0] = 1

        # Right boundary constraint
        D[ndof + 1, -1] = 1
        D[ndof + 3, -1] = 1

        # Additional constraints
        D[ndof + 2:, ndof + 2:] = DD

        rhs = np.concatenate([
            -S @ h - M @ (2 * g2 * h - g1 * x),
            [SL + (h[0] ** 2) / 2],
            [-(SR + (h[-1] ** 2) / 2)],
            [0, 0]
        ])

        try:
            u = spsolve(D.tocsr(), rhs)
        except Exception as e:
            print("Error solving system:", e)
            print("D matrix:", D)
            print("rhs:", rhs)
            break

        U = spsolve(I - P, u[:len(x)])
        h += dt * U
        x += dt * (X @ U)
        t += dt

    plt.plot(x, h, label=f"t={t:.2f}")

    plt.xlabel("x")
    plt.ylabel("h")
    plt.legend()
    plt.title("Thin Film Evolution")
    plt.savefig("b.png")

if __name__ == "__main__":
    thin_film_simulation()

