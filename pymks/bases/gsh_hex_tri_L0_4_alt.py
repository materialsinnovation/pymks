import numpy as np


def gsh_basis_info():

    indxvec = np.array([[0, 0, 1],
                        [2, -2, 1],
                        [2, -1, 1],
                        [2, 0, 1],
                        [2, 1, 1],
                        [2, 2, 1],
                        [4, -4, 1],
                        [4, -3, 1],
                        [4, -2, 1],
                        [4, -1, 1],
                        [4, 0, 1],
                        [4, 1, 1],
                        [4, 2, 1],
                        [4, 3, 1],
                        [4, 4, 1]])

    return indxvec


def gsh_eval(X, Bvec):

    phi1 = X[..., 0]
    phi = X[..., 1]
    phi2 = X[..., 2]

    zvec = np.abs(phi) < 1e-8
    zvec = zvec.astype(int)
    # randvec = np.round(np.random.rand(zvec.shape[0]))
    # randvecopp = np.ones(zvec.shape[0]) - randvec
    randvec = np.round(np.random.rand(zvec.size)).reshape(zvec.shape)
    randvecopp = np.ones(zvec.shape) - randvec
    phi += (1e-7)*zvec*(randvec - randvecopp)

    final_shape = np.hstack([phi1.shape, len(Bvec)])  # shape of the output data
    out_tvalues = np.zeros(final_shape, dtype = 'complex128')

    c = 0
    for Bval in Bvec:

        if Bval == 0:
            out_tvalues[..., c] = 1

        if Bval == 1:
            t183 = np.sin(phi)
            out_tvalues[..., c] = -(0.5e1 / 0.4e1) * np.exp((-2*1j) * phi1) * np.sqrt(0.6e1) * t183 ** 2

        if Bval == 2:
            t184 = np.cos(phi)
            out_tvalues[..., c] = (-0.5e1 / 0.2e1*1j) * np.exp((-1*1j) * phi1) * np.sqrt(0.6e1) * np.sqrt((1 + t184)) * t184 * np.sqrt((1 - t184))

        if Bval == 3:
            t185 = np.cos(phi)
            out_tvalues[..., c] = 0.15e2 / 0.2e1 * t185 ** 2 - 0.5e1 / 0.2e1

        if Bval == 4:
            t186 = np.cos(phi)
            out_tvalues[..., c] = (-0.5e1 / 0.2e1*1j) * np.exp((1j) * phi1) * np.sqrt(0.6e1) * np.sqrt((1 - t186)) * np.sqrt((1 + t186)) * t186

        if Bval == 5:
            t187 = np.sin(phi)
            out_tvalues[..., c] = -(0.5e1 / 0.4e1) * np.exp((2*1j) * phi1) * np.sqrt(0.6e1) * t187 ** 2

        if Bval == 6:
            t190 = np.sin(phi)
            t188 = t190 ** 2
            out_tvalues[..., c] = (0.9e1 / 0.16e2) * np.exp((-4*1j) * phi1) * np.sqrt(0.70e2) * t188 ** 2

        if Bval == 7:
            t191 = np.cos(phi)
            out_tvalues[..., c] = (0.9e1 / 0.4e1*1j) * t191 * (1 + (-2 + t191) * t191) * ((1 + t191) ** (0.3e1 / 0.2e1)) * np.sqrt(0.35e2) * np.exp((-3*1j) * phi1) * ((1 - t191) ** (-0.1e1 / 0.2e1))

        if Bval == 8:
            t193 = np.cos(phi)
            t192 = np.sin(phi)
            out_tvalues[..., c] = -(0.9e1 / 0.8e1) * np.exp((-2*1j) * phi1) * np.sqrt(0.10e2) * t192 ** 2 * (7 * t193 ** 2 - 1)

        if Bval == 9:
            t194 = np.cos(phi)
            out_tvalues[..., c] = (-0.9e1 / 0.4e1*1j) * np.exp((-1*1j) * phi1) * np.sqrt(0.5e1) * np.sqrt((1 + t194)) * t194 * np.sqrt((1 - t194)) * (7 * t194 ** 2 - 3)

        if Bval == 10:
            t199 = np.cos(phi)
            t200 = t199 ** 2
            out_tvalues[..., c] = 0.27e2 / 0.8e1 + (-0.135e3 / 0.4e1 + 0.315e3 / 0.8e1 * t200) * t200

        if Bval == 11:
            t202 = np.cos(phi)
            out_tvalues[..., c] = (-0.9e1 / 0.4e1*1j) * np.exp((1j) * phi1) * np.sqrt(0.5e1) * np.sqrt((1 - t202)) * np.sqrt((1 + t202)) * t202 * (7 * t202 ** 2 - 3)

        if Bval  == 12:
            t204 = np.cos(phi)
            t203 = np.sin(phi)
            out_tvalues[..., c] = -(0.9e1 / 0.8e1) * np.exp((2*1j) * phi1) * np.sqrt(0.10e2) * t203 ** 2 * (7 * t204 ** 2 - 1)

        if Bval  == 13:
            t205 = np.cos(phi)
            out_tvalues[..., c] = (0.9e1 / 0.4e1*1j) * np.exp((3*1j) * phi1) * np.sqrt(0.35e2) * ((1 - t205) ** (0.3e1 / 0.2e1)) * ((1 + t205) ** (0.3e1 / 0.2e1)) * t205

        if Bval  == 14:
            t208 = np.sin(phi)
            t206 = t208 ** 2
            out_tvalues[..., c] = (0.9e1 / 0.16e2) * np.exp((4*1j) * phi1) * np.sqrt(0.70e2) * t206 ** 2

        c += 1

    return out_tvalues


if __name__ == '__main__':

    X = np.zeros([2, 3])
    phi1 = np.array([0.1,0.2])
    X[:, 0] = phi1
    phi = np.array([0.0, 0.4])
    X[:, 1] = phi
    phi2 = np.array([0.3, 0.6])
    X[:, 2] = phi2

    indxvec = gsh_basis_info()
    print indxvec

    lte2 = indxvec[:, 0] <= 2
    print lte2

    Bvec = np.arange(indxvec.shape[0])[lte2]
    print Bvec

    out_tvalues = gsh_eval(X, Bvec)
    print out_tvalues
    print out_tvalues.shape
