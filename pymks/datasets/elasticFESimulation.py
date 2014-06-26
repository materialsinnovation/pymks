import numpy as np
from sfepy.base.goptions import goptions
goptions['verbose'] = False
from sfepy.base.base import IndexedStruct
from sfepy.discrete.fem import Field
try:
    from sfepy.discrete.fem import FEDomain as Domain
except ImportError:
    from sfepy.discrete.fem import Domain
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
import sfepy.discrete.fem.periodic as per
from sfepy.discrete import Functions
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.mechanics.matcoefs import ElasticConstants
from sfepy.base.base import output
output.set_output(quiet=True)

class ElasticFESimulation(object):
    """

    Use SfePy to solve a linear strain problem in 2D with a varying
    microstructure on a rectangular grid. The rectangle (cube) is held
    at the negative edge (plane) and displaced by 1 on the positive x
    edge (plane). Periodic boundary conditions are applied to the
    other boundaries.

    The microstructure is of shape (Nsample, Nx, Ny) or (Nsample, Nx, Ny, Nz).

    >>> X = np.zeros((1, 3, 3), dtype=int)
    >>> X[0, :, 1] = 1

    >>> model = ElasticFESimulation(elastic_modulus=(1.0, 10.0), poissons_ratio=(0., 0.))
    >>> y = model.get_response(X, slice(None)) # doctest: +ELLIPSIS
    sfepy: ...

    y is the strain with components as follows

    >>> exx = y[..., 0]
    >>> eyy = y[..., 1]
    >>> exy = y[..., 2]

    Since there is no contrast in the microstructure the strain is only
    in the x-direction and has a uniform value of 1 since the
    displacement is always 1 and the size of the domain is 1.

    >>> assert np.allclose(exx, 1)
    >>> assert np.allclose(eyy, 0)
    >>> assert np.allclose(exy, 0)

    """
    def __init__(self, elastic_modulus, poissons_ratio, strain=1.,):
        """
        Args:
          elastic_modulus: array of elastic moduli for phases
          poissons_ratio: array of possion ratios for phases
          strain: Scalar for macroscopic strain

        """
        self.strain = strain
        self.dx = 1.0
        self.elastic_modulus = elastic_modulus
        self.poissons_ratio = poissons_ratio
        if len(elastic_modulus) != len(poissons_ratio):
            raise RuntimeError, 'elastic_modulus and poissons_ratio must be the same length'

    def _convert_properties(self, dim):
        """
        Convert from elastic modulus and Poisson's ratio to the Lame
        parameter and shear modulus

        >>> model = ElasticFESimulation(elastic_modulus=(1., 2.), poissons_ratio=(1., 1.))
        >>> result = model._convert_properties(2)
        >>> answer = np.array([[-0.5, 1. / 6.], [-1., 1. / 3.]])
        >>> assert(np.allclose(result, answer))

        Args:
           dim: Scalar value for the dimension of the microstructure.

        Returns:
            array with the Lame parameter and the shear modulus for each phase.

        """
        def _convert(E, nu):
            ec = ElasticConstants(young=E, poisson=nu)
            mu = dim / 3. * ec.mu
            lame = ec.lam
            return lame, mu

        return np.array([_convert(E, nu) for E, nu in zip(self.elastic_modulus, self.poissons_ratio)])

    def _get_property_array(self, X):
        """
        Generate property array with elastic_modulus and poissons_ratio for each phase.

        Test case for 2D with 3 phases.

        >>> X2D = np.array([[[0, 1, 2, 1],
        ...                  [2, 1, 0, 0],
        ...                  [1, 0, 2, 2]]])
        >>> model2D = ElasticFESimulation(elastic_modulus=(1., 2., 3.), poissons_ratio=(1., 1., 1.))
        >>> lame = lame0, lame1, lame2 = -0.5, -1., -1.5
        >>> mu = mu0, mu1, mu2 = 1. / 6, 1. / 3, 1. / 2
        >>> lm = zip(lame, mu)
        >>> X2D_property = np.array([[lm[0], lm[1], lm[2], lm[1]],
        ...                          [lm[2], lm[1], lm[0], lm[0]],
        ...                          [lm[1], lm[0], lm[2], lm[2]]])

        >>> assert(np.allclose(model2D._get_property_array(X2D), X2D_property))

        Test case for 3D with 2 phases.

        >>> model3D = ElasticFESimulation(elastic_modulus=(1., 2.), poissons_ratio=(1., 1.))
        >>> X3D = np.array([[[0, 1],
        ...                  [0, 0]],
        ...                 [[1, 1],
        ...                  [0, 1]]])
        >>> X3D_property = np.array([[[lm[0], lm[1]],
        ...                           [lm[0], lm[0]]],
        ...                          [[lm[1], lm[1]],
        ...                           [lm[0], lm[1]]]])
        >>> assert(np.allclose(model3D._get_property_array(X3D), X3D_property))

        """
        dim = len(X.shape) - 1
        Nphase = len(self.elastic_modulus)
        if not issubclass(X.dtype.type, np.integer):
            raise TypeError, "X must be an integer array"
        if Nphase != np.max(X) + 1:
            raise RuntimeError, "X has the wrong number of phases."
        if np.min(X) != 0:
            raise RuntimeError, "Phases must be zero indexed."
        if not (2 <= dim <= 3):
            raise RuntimeError, "the shape of X is incorrect"
        return self._convert_properties(dim)[X]

    def get_response(self, X, strain_index=0):
        """
        Get the strain fields given an initial microstructure
        with a macroscopic strain applied in the x direction.

        Args:
          X: microstructure with shape (Nsample, Nx, Ny) or
             (Nsample, Nx, Ny, Nz) 
          strain_index: interger value to return
             a particular strain field.  0 returns exx, 1 returns eyy,
             etc. To return all strain fields set `strain_index` equal to
             `slice(None)`.

        Returns:
          the strain fields over each cell

        """
        X_property = self._get_property_array(X)
        y_strain = np.array([self._solve(x) for x in X_property])
        return y_strain[..., strain_index]

    def _get_material(self, property_array, domain):
        """
        Creates an SfePy material from the material property fields for the quadrature points.

        Args:
          property_array: array of the properties with shape (Nx, Ny, Nz, 2)

        Returns:
          an SfePy material

        """
        min_xyz = domain.get_mesh_bounding_box()[0]

        def _material_func_(ts, coors, mode=None, **kwargs):
            if mode == 'qp':
                ijk_out = np.empty_like(coors, dtype=int)
                ijk = np.floor((coors - min_xyz[None]) / self.dx, ijk_out)
                ijk_tuple = tuple(ijk.swapaxes(0, 1))
                property_array_qp = property_array[ijk_tuple]
                lam = property_array_qp[..., 0]
                mu = property_array_qp[..., 1]
                lam = np.ascontiguousarray(lam.reshape((lam.shape[0], 1, 1)))
                mu = np.ascontiguousarray(mu.reshape((mu.shape[0], 1, 1)))
                return {'lam' : lam, 'mu' : mu}
            else:
                return

        material_func = Function('material_func', _material_func_)
        return Material('m', function=material_func)

    def _subdomain_func(self, x=(), y=(), z=()):
        """
        Creates a function to mask subdomains in Sfepy.

        Args:
          x: tuple of lines or points to be masked in the x-plane
          y: tuple of lines or points to be masked in the y-plane
          z: tuple of lines or points to be masked in the z-plane

        Returns:
          array of masked location indices

        """
        eps = 1e-3 * self.dx

        def _func(coords, domain=None):
            flag_x = len(x) == 0
            flag_y = len(y) == 0
            flag_z = len(z) == 0
            for x_ in x:
                flag = (coords[:, 0] < (x_ + eps)) & (coords[:, 0] > (x_ - eps))
                flag_x = flag_x | flag
            for y_ in y:
                flag = (coords[:, 1] < (y_ + eps)) & (coords[:, 1] > (y_ - eps))
                flag_y = flag_y | flag
            for z_ in z:
                flag = (coords[:, 2] < (z_ + eps)) & (coords[:, 2] > (z_ - eps))
                flag_z = flag_z | flag
            return np.where(flag_x & flag_y & flag_z)[0]

        return _func

    def _get_periodicBC(self, domain, dim):
        dim_dict = {1 : ('y', per.match_y_plane),
                       2 : ('z', per.match_z_plane)}
        dim_string = dim_dict[dim][0]
        match_plane = dim_dict[dim][1]
        min_, max_ = domain.get_mesh_bounding_box()[:, dim]
        plus_ = self._subdomain_func(**{dim_string : (max_,)})
        minus_ = self._subdomain_func(**{dim_string: (min_,)})
        plus_string = dim_string + 'plus'
        minus_string = dim_string + 'minus'
        plus = Function(plus_string, plus_)
        minus = Function(minus_string, minus_)
        region_plus = domain.create_region('region_{0}_plus'.format(dim_string),
                                           'vertices by {0}'.format(plus_string),
                                           'facet',
                                           functions=Functions([plus]))
        region_minus = domain.create_region('region_{0}_minus'.format(dim_string),
                                            'vertices by {0}'.format(minus_string),
                                            'facet',
                                            functions=Functions([minus]))
        match_plane = Function('match_{0}_plane'.format(dim_string), match_plane)
        bc = PeriodicBC('periodic_{0}'.format(dim_string),
                        [region_plus, region_minus],
                        {'u.all' : 'u.all'},
                        match='match_{0}_plane'.format(dim_string))
        return bc, match_plane

    def _get_periodicBCs(self, domain):
        dims = domain.get_mesh_bounding_box().shape[1]

        bc_list, func_list = zip(*[self._get_periodicBC(domain, i) for i in range(1, dims)])
        return Conditions(bc_list), Functions(func_list)

    def _get_displacementBCs(self, domain):
        """
        Fix the left plane in x, displace the right plane by 1 and fix
        the y-direction with the top and bottom points on the left x
        plane.

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        max_xyz = domain.get_mesh_bounding_box()[1]
        xplus_ = self._subdomain_func(x=(max_xyz[0],))
        xminus_ = self._subdomain_func(x=(min_xyz[0],))

        kwargs = {}
        if len(min_xyz) == 3:
            kwargs = {'z' : (max_xyz[2], min_xyz[2])}
        fix_x_points_ = self._subdomain_func(x=(min_xyz[0],),
                                             y=(max_xyz[1], min_xyz[1]),
                                             **kwargs)

        xplus = Function('xplus', xplus_)
        xminus = Function('xminus', xminus_)
        fix_x_points = Function('fix_x_points', fix_x_points_)
        region_x_plus = domain.create_region('region_x_plus',
                                            'vertices by xplus',
                                            'facet',
                                             functions=Functions([xplus]))
        region_left = domain.create_region('region_x_minus',
                                           'vertices by xminus',
                                           'facet',
                                            functions=Functions([xminus]))
        region_fix_points = domain.create_region('region_fix_points',
                                          'vertices by fix_x_points',
                                          'vertex',
                                           functions=Functions([fix_x_points]))
        fixed_BC = EssentialBC('fixed_BC', region_left, {'u.0' : 0.0})
        displaced_BC = EssentialBC('displaced_BC', region_x_plus,
                                   {'u.0' : self.strain * (max_xyz[0] - min_xyz[0])})
        fix_points_BC = EssentialBC('fix_points_BC', region_fix_points, {'u.1' : 0.0})
        return Conditions([fixed_BC, displaced_BC, fix_points_BC])

    def _get_mesh(self, shape):
        """
        Generate an Sfepy rectangular mesh

        Args:
          shape: proposed shape of domain (vertex shape) (Nx, Ny)

        Returns:
          Sfepy mesh

        """
        center = np.zeros_like(shape)
        return gen_block_mesh(shape, np.array(shape) + 1, center, verbose=False)

    def _solve(self, property_array):
        """
        Solve the Sfepy problem for one sample.

        Args:
          property_array: array of shape (Nx, Ny, 2) where the last
          index is for Lame's parameter and shear modulus,
          respectively.

        Returns:
          the strain field of shape (Nx, Ny, 2) where the last
          index represents the x and y displacements

        """
        shape = property_array.shape[:-1]
        mesh = self._get_mesh(shape)
        domain = Domain('domain', mesh)

        region_all = domain.create_region('region_all', 'all')

        field = Field.from_args('fu', np.float64, 'vector', region_all, approx_order=2)

        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        m = self._get_material(property_array, domain)

        integral = Integral('i', order=3)

        t1 = Term.new('dw_lin_elastic_iso(m.lam, m.mu, v, u)',
                      integral, region_all, m=m, v=v, u=u)
        eq = Equation('balance_of_forces', t1)
        eqs = Equations([eq])

        ls = ScipyDirect({})
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)

        pb = Problem('elasticity', equations=eqs, nls=nls, ls=ls)
        pb.save_regions_as_groups('regions')

        epbcs, functions = self._get_periodicBCs(domain)
        ebcs = self._get_displacementBCs(domain)

        pb.time_update(ebcs=ebcs, epbcs=epbcs, functions=functions)
        pb.solve()

        strain = np.squeeze(pb.evaluate('ev_cauchy_strain.3.region_all(u)', mode='el_avg'))
        return np.reshape(strain, (shape + strain.shape[-1:]))
