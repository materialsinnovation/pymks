import numpy as np
from pymks.bases import GSHBasis


def test_gsh_no_symmetry():
    """this test checks that a particular gsh basis function with no
    symmetry is being evaluated properly"""

    X = np.array([[0.1, 0.2, 0.3],
                  [6.5, 2.3, 3.4]])
    gsh_basis = GSHBasis(n_states=[1])

    assert(np.allclose(np.squeeze(gsh_basis.discretize(X)), q_no_symm(X)))


def test_gsh_hex():
    """this test checks that a particular gsh basis function for hexagonal
    symmetry is being evaluated properly"""

    X = np.array([[0.1, 0.2, 0.3],
                  [6.5, 2.3, 3.4]])
    gsh_basis = GSHBasis(n_states=[1], domain='hexagonal')

    assert(np.allclose(np.squeeze(gsh_basis.discretize(X)), q_hex(X)))


def test_symmetry_check_hex():
    """this test is designed to check that the hexagonal gsh functions
    for two symmetrically equivalent orientations output the same gsh
    coefficients"""

    X1 = np.array([[30, 70, 45]])*np.pi/180.
    X2 = np.array([[30+180, 180-70, 2*60-45]])*np.pi/180.
    gsh_basis = GSHBasis(n_states=np.arange(0, 100, 5), domain='hexagonal')

    assert(np.allclose(gsh_basis.discretize(X1), gsh_basis.discretize(X2)))


def q_hex(x):
    phi1 = x[:, 0]
    phi = x[:, 1]
    t913 = np.sin(phi)
    x_GSH = -((0.5e1 / 0.4e1) * np.exp((-2*1j) * phi1) *
              np.sqrt(0.6e1) * t913 ** 2)
    return x_GSH


def q_no_symm(x):
    phi1 = x[:, 0]
    phi = x[:, 1]
    phi2 = x[:, 2]
    x_GSH = ((0.3e1 / 0.2e1) * (0.1e1 + np.cos(phi)) *
             np.exp((-1*1j) * (phi1 + phi2)))
    return x_GSH


if __name__ == '__main__':
    test_gsh_no_symmetry()
    test_gsh_hex()
    test_symmetry_check_hex()
