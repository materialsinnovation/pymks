"""Test the primitive basis.
"""

import numpy as np
from pymks.fmks.bases import legendre as leg


def polyval(x_data):
    """
        Evaluate Legendre expansion for given input.
    """
    x_data = 4 * x_data - 1
    polys = np.array((np.ones_like(x_data), x_data, (3. * x_data ** 2 - 1.) / 2.))
    temp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
    return np.rollaxis(temp, 0, 3)


def test_1():
    """
        This test compares output using fmks Legendre
        discretization utility to independantly evaluated
        polynomial expansion results.
    """

    n_state = 3
    data = np.array([[0.25, 0.1], [0.5, 0.25]])
    domain = [0., 0.5]
    chunks = (1,)
    assert (
        np.allclose(
            leg.legendre_basis(data, n_state, domain, chunks)[0].compute(),
            polyval(data),
        )
    )
    n_state = 3
    data = np.array([[-1, 1], [0, -1]])
    domain = [0., 0.5]
    chunks = (1,)
    assert (
        np.allclose(
            leg.legendre_basis(data, n_state, domain, chunks)[0].compute(),
            polyval(data),
        )
    )


def test_2():
    """
    This test compares output from Legendre discretization to
    known, hardcoded results.
    """
    np.random.seed(3)
    data = np.random.random((1, 3, 3))
    data_ = leg.legendre_basis(data, n_state=2, domain=(0, 1))[0].compute()
    f_data = np.fft.fftn(data_, axes=(1, 2))
    f_test = np.array(
        [
            [
                [
                    [4.50000000 + 0.j, -0.79735949 + 0.],
                    [0.00000000 + 0.j, -1.00887157 - 1.48005289j],
                    [0.00000000 + 0.j, -1.00887157 + 1.48005289j],
                ],
                [
                    [0.00000000 + 0.j, 0.62300683 - 4.97732233j],
                    [0.00000000 + 0.j, 1.09318216 + 0.10131035j],
                    [0.00000000 + 0.j, 0.37713401 + 1.87334545j],
                ],
                [
                    [0.00000000 + 0.j, 0.62300683 + 4.97732233j],
                    [0.00000000 + 0.j, 0.37713401 - 1.87334545j],
                    [0.00000000 + 0.j, 1.09318216 - 0.10131035j],
                ],
            ]
        ]
    )
    assert np.allclose(f_data, f_test)
