try:
    import sfepy
except ImportError:
    import pytest
    pytest.importorskip('sfepy')
    raise

import numpy as np
from sfepy.base.goptions import goptions
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
from sfepy.discrete.conditions import LinearCombinationBC

goptions['verbose'] = False
output.set_output(quiet=True)


class ElasticFESimulation(object):

    """
    Use SfePy to solve a linear strain problem in 2D with a varying
    microstructure on a rectangular grid. The rectangle (cube) is held
    at the negative edge (plane) and displaced by 1 on the positive x
    edge (plane). Periodic boundary conditions are applied to the
    other boundaries.

    The microstructure is of shape (n_samples, n_x, n_y) or (n_samples, n_x,
    n_y, n_z).

    >>> X = np.zeros((1, 3, 3), dtype=int)
    >>> X[0, :, 1] = 1

    >>> sim = ElasticFESimulation(elastic_modulus=(1.0, 10.0),
    ...                           poissons_ratio=(0., 0.))
    >>> sim.run(X)
    >>> y = sim.strain

    y is the strain with components as follows

    >>> exx = y[..., 0]
    >>> eyy = y[..., 1]
    >>> exy = y[..., 2]

    In this example, the strain is only in the x-direction and has a
    uniform value of 1 since the displacement is always 1 and the size
    of the domain is 1.

    >>> assert np.allclose(exx, 1)
    >>> assert np.allclose(eyy, 0)
    >>> assert np.allclose(exy, 0)

    The following example is for a system with contrast. It tests the
    left/right periodic offset and the top/bottom periodicity.

    >>> X = np.array([[[1, 0, 0, 1],
    ...                [0, 1, 1, 1],
    ...                [0, 0, 1, 1],
    ...                [1, 0, 0, 1]]])
    >>> n_samples, N, N = X.shape
    >>> macro_strain = 0.1
    >>> sim = ElasticFESimulation((10.0,1.0), (0.3,0.3), macro_strain=0.1)
    >>> sim.run(X)
    >>> u = sim.displacement[0]

    Check that the offset for the left/right planes is `N *
    macro_strain`.

    >>> assert np.allclose(u[-1,:,0] - u[0,:,0], N * macro_strain)

    Check that the left/right side planes are periodic in y.

    >>> assert np.allclose(u[0,:,1], u[-1,:,1])

    Check that the top/bottom planes are periodic in both x and y.

    >>> assert np.allclose(u[:,0], u[:,-1])

    """

    def __init__(self, elastic_modulus, poissons_ratio, macro_strain=1.,):
        """Instantiate a ElasticFESimulation.

        Args:
            elastic_modulus (1D array): array of elastic moduli for phases
            poissons_ratio (1D array): array of Possion's ratios for phases
            macro_strain (float, optional): Scalar for macroscopic strain

        """
        self.macro_strain = macro_strain
        self.dx = 1.0
        self.elastic_modulus = elastic_modulus
        self.poissons_ratio = poissons_ratio
        if len(elastic_modulus) != len(poissons_ratio):
            raise RuntimeError(
                'elastic_modulus and poissons_ratio must be the same length')

    def _convert_properties(self, dim):
        """
        Convert from elastic modulus and Poisson's ratio to the Lame
        parameter and shear modulus

        >>> model = ElasticFESimulation(elastic_modulus=(1., 2.),
        ...                             poissons_ratio=(1., 1.))
        >>> result = model._convert_properties(2)
        >>> answer = np.array([[-0.5, 1. / 6.], [-1., 1. / 3.]])
        >>> assert(np.allclose(result, answer))

        Args:
            dim (int): Scalar value for the dimension of the microstructure.

        Returns:
            array with the Lame parameter and the shear modulus for each phase.

        """
        def _convert(E, nu):
            ec = ElasticConstants(young=E, poisson=nu)
            mu = dim / 3. * ec.mu
            lame = ec.lam
            return lame, mu

        return np.array([_convert(E, nu) for E,
                         nu in zip(self.elastic_modulus, self.poissons_ratio)])

    def _get_property_array(self, X):
        """
        Generate property array with elastic_modulus and poissons_ratio for
        each phase.

        Test case for 2D with 3 phases.

        >>> X2D = np.array([[[0, 1, 2, 1],
        ...                  [2, 1, 0, 0],
        ...                  [1, 0, 2, 2]]])
        >>> model2D = ElasticFESimulation(elastic_modulus=(1., 2., 3.),
        ...                               poissons_ratio=(1., 1., 1.))
        >>> lame = lame0, lame1, lame2 = -0.5, -1., -1.5
        >>> mu = mu0, mu1, mu2 = 1. / 6, 1. / 3, 1. / 2
        >>> lm = zip(lame, mu)
        >>> X2D_property = np.array([[lm[0], lm[1], lm[2], lm[1]],
        ...                          [lm[2], lm[1], lm[0], lm[0]],
        ...                          [lm[1], lm[0], lm[2], lm[2]]])

        >>> assert(np.allclose(model2D._get_property_array(X2D), X2D_property))

        Test case for 3D with 2 phases.

        >>> model3D = ElasticFESimulation(elastic_modulus=(1., 2.),
        ...                               poissons_ratio=(1., 1.))
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
        n_phases = len(self.elastic_modulus)
        if not issubclass(X.dtype.type, np.integer):
            raise TypeError("X must be an integer array")
        if np.max(X) >= n_phases or np.min(X) < 0:
            raise RuntimeError(
                "X must be between 0 and {N}.".format(N=n_phases - 1))
        if not (2 <= dim <= 3):
            raise RuntimeError("the shape of X is incorrect")
        return self._convert_properties(dim)[X]

    def run(self, X):
        """
        Run the simulation.

        Args:
          X (ND array): microstructure with shape (n_samples, n_x, ...)
        """
        X_property = self._get_property_array(X)
        strain = []
        displacement = []
        stress = []
        for x in X_property:
            strain_, displacement_, stress_ = self._solve(x)
            strain.append(strain_)
            displacement.append(displacement_)
            stress.append(stress_)
        self.strain = np.array(strain)
        self.displacement = np.array(displacement)
        self.stress = np.array(stress)

    @property
    def response(self):
        return self.strain[..., 0]

    def _get_material(self, property_array, domain):
        """
        Creates an SfePy material from the material property fields for the
        quadrature points.

        Args:
          property_array: array of the properties with shape (n_x, n_y, n_z, 2)

        Returns:
          an SfePy material

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        dims = domain.get_mesh_bounding_box().shape[1]

        def _material_func_(ts, coors, mode=None, **kwargs):
            if mode == 'qp':
                ijk_out = np.empty_like(coors, dtype=int)
                ijk = np.floor((coors - min_xyz[None]) / self.dx,
                               ijk_out, casting="unsafe")
                ijk_tuple = tuple(ijk.swapaxes(0, 1))
                property_array_qp = property_array[ijk_tuple]
                lam = property_array_qp[..., 0]
                mu = property_array_qp[..., 1]
                lam = np.ascontiguousarray(lam.reshape((lam.shape[0], 1, 1)))
                mu = np.ascontiguousarray(mu.reshape((mu.shape[0], 1, 1)))

                from sfepy.mechanics.matcoefs import stiffness_from_lame
                stiffness = stiffness_from_lame(dims, lam=lam, mu=mu)
                return {'lam': lam, 'mu': mu, 'D': stiffness}
            else:
                return

        material_func = Function('material_func', _material_func_)
        return Material('m', function=material_func)

    def _subdomain_func(self, x=(), y=(), z=(), max_x=None):
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
                flag = (coords[:, 0] < (x_ + eps)) & \
                       (coords[:, 0] > (x_ - eps))
                flag_x = flag_x | flag
            for y_ in y:
                flag = (coords[:, 1] < (y_ + eps)) & \
                       (coords[:, 1] > (y_ - eps))
                flag_y = flag_y | flag
            for z_ in z:
                flag = (coords[:, 2] < (z_ + eps)) & \
                       (coords[:, 2] > (z_ - eps))
                flag_z = flag_z | flag
            flag = flag_x & flag_y & flag_z
            if max_x is not None:
                flag = flag & (coords[:, 0] < (max_x - eps))
            return np.where(flag)[0]

        return _func

    def _get_mesh(self, shape):
        """
        Generate an Sfepy rectangular mesh

        Args:
          shape: proposed shape of domain (vertex shape) (n_x, n_y)

        Returns:
          Sfepy mesh

        """
        center = np.zeros_like(shape)
        return gen_block_mesh(shape, np.array(shape) + 1, center,
                              verbose=False)

    def _get_fixed_displacementsBCs(self, domain):
        """
        Fix the left top and bottom points in x, y and z

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        max_xyz = domain.get_mesh_bounding_box()[1]

        kwargs = {}
        fix_points_dict = {'u.0': 0.0, 'u.1': 0.0}
        if len(min_xyz) == 3:
            kwargs = {'z': (max_xyz[2], min_xyz[2])}
            fix_points_dict['u.2'] = 0.0
        fix_x_points_ = self._subdomain_func(x=(min_xyz[0],),
                                             y=(max_xyz[1], min_xyz[1]),
                                             **kwargs)

        fix_x_points = Function('fix_x_points', fix_x_points_)
        region_fix_points = domain.create_region(
            'region_fix_points',
            'vertices by fix_x_points',
            'vertex',
            functions=Functions([fix_x_points]))
        return EssentialBC('fix_points_BC', region_fix_points, fix_points_dict)

    def _get_shift_displacementsBCs(self, domain):
        """
        Fix the right top and bottom points in x, y and z

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        max_xyz = domain.get_mesh_bounding_box()[1]
        kwargs = {}
        if len(min_xyz) == 3:
            kwargs = {'z': (max_xyz[2], min_xyz[2])}

        displacement = self.macro_strain * (max_xyz[0] - min_xyz[0])
        shift_points_dict = {'u.0': displacement}

        shift_x_points_ = self._subdomain_func(x=(max_xyz[0],),
                                               y=(max_xyz[1], min_xyz[1]),
                                               **kwargs)

        shift_x_points = Function('shift_x_points', shift_x_points_)
        region_shift_points = domain.create_region(
            'region_shift_points',
            'vertices by shift_x_points',
            'vertex',
            functions=Functions([shift_x_points]))
        return EssentialBC('shift_points_BC',
                           region_shift_points, shift_points_dict)

    def _get_displacementBCs(self, domain):
        shift_points_BC = self._get_shift_displacementsBCs(domain)
        fix_points_BC = self._get_fixed_displacementsBCs(domain)
        return Conditions([fix_points_BC, shift_points_BC])

    def _get_linear_combinationBCs(self, domain):
        """
        The right nodes are periodic with the left nodes but also displaced.

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        min_xyz = domain.get_mesh_bounding_box()[0]
        max_xyz = domain.get_mesh_bounding_box()[1]
        xplus_ = self._subdomain_func(x=(max_xyz[0],))
        xminus_ = self._subdomain_func(x=(min_xyz[0],))

        xplus = Function('xplus', xplus_)
        xminus = Function('xminus', xminus_)
        region_x_plus = domain.create_region('region_x_plus',
                                             'vertices by xplus',
                                             'facet',
                                             functions=Functions([xplus]))
        region_x_minus = domain.create_region('region_x_minus',
                                              'vertices by xminus',
                                              'facet',
                                              functions=Functions([xminus]))
        match_x_plane = Function('match_x_plane', per.match_x_plane)

        def shift_(ts, coors, region):
            return np.ones_like(coors[:, 0]) * \
                self.macro_strain * (max_xyz[0] - min_xyz[0])
        shift = Function('shift', shift_)
        lcbc = LinearCombinationBC(
            'lcbc', [region_x_plus, region_x_minus], {
                'u.0': 'u.0'}, match_x_plane, 'shifted_periodic',
            arguments=(shift,))

        return Conditions([lcbc])

    def _get_periodicBC_X(self, domain, dim):
        dim_dict = {1: ('y', per.match_y_plane),
                    2: ('z', per.match_z_plane)}
        dim_string = dim_dict[dim][0]
        match_plane = dim_dict[dim][1]
        min_, max_ = domain.get_mesh_bounding_box()[:, dim]
        min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
        plus_ = self._subdomain_func(max_x=max_x, **{dim_string: (max_,)})
        minus_ = self._subdomain_func(max_x=max_x, **{dim_string: (min_,)})
        plus_string = dim_string + 'plus'
        minus_string = dim_string + 'minus'
        plus = Function(plus_string, plus_)
        minus = Function(minus_string, minus_)
        region_plus = domain.create_region(
            'region_{0}_plus'.format(dim_string),
            'vertices by {0}'.format(
                plus_string),
            'facet',
            functions=Functions([plus]))
        region_minus = domain.create_region(
            'region_{0}_minus'.format(dim_string),
            'vertices by {0}'.format(
                minus_string),
            'facet',
            functions=Functions([minus]))
        match_plane = Function(
            'match_{0}_plane'.format(dim_string), match_plane)

        bc_dict = {'u.0': 'u.0'}

        bc = PeriodicBC('periodic_{0}'.format(dim_string),
                        [region_plus, region_minus],
                        bc_dict,
                        match='match_{0}_plane'.format(dim_string))
        return bc, match_plane

    def _get_periodicBC_YZ(self, domain, dim):
        dims = domain.get_mesh_bounding_box().shape[1]
        dim_dict = {0: ('x', per.match_x_plane),
                    1: ('y', per.match_y_plane),
                    2: ('z', per.match_z_plane)}
        dim_string = dim_dict[dim][0]
        match_plane = dim_dict[dim][1]
        min_, max_ = domain.get_mesh_bounding_box()[:, dim]
        plus_ = self._subdomain_func(**{dim_string: (max_,)})
        minus_ = self._subdomain_func(**{dim_string: (min_,)})
        plus_string = dim_string + 'plus'
        minus_string = dim_string + 'minus'
        plus = Function(plus_string, plus_)
        minus = Function(minus_string, minus_)
        region_plus = domain.create_region(
            'region_{0}_plus'.format(dim_string),
            'vertices by {0}'.format(
                plus_string),
            'facet',
            functions=Functions([plus]))
        region_minus = domain.create_region(
            'region_{0}_minus'.format(dim_string),
            'vertices by {0}'.format(
                minus_string),
            'facet',
            functions=Functions([minus]))
        match_plane = Function(
            'match_{0}_plane'.format(dim_string), match_plane)

        bc_dict = {'u.1': 'u.1'}
        if dims == 3:
            bc_dict['u.2'] = 'u.2'

        bc = PeriodicBC('periodic_{0}'.format(dim_string),
                        [region_plus, region_minus],
                        bc_dict,
                        match='match_{0}_plane'.format(dim_string))
        return bc, match_plane

    def _solve(self, property_array):
        """
        Solve the Sfepy problem for one sample.

        Args:
          property_array: array of shape (n_x, n_y, 2) where the last
          index is for Lame's parameter and shear modulus,
          respectively.

        Returns:
          the strain field of shape (n_x, n_y, 2) where the last
          index represents the x and y displacements

        """
        shape = property_array.shape[:-1]
        mesh = self._get_mesh(shape)
        domain = Domain('domain', mesh)

        region_all = domain.create_region('region_all', 'all')

        field = Field.from_args('fu', np.float64, 'vector', region_all,
                                approx_order=2)

        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        m = self._get_material(property_array, domain)

        integral = Integral('i', order=4)

        t1 = Term.new('dw_lin_elastic_iso(m.lam, m.mu, v, u)',
                      integral, region_all, m=m, v=v, u=u)
        eq = Equation('balance_of_forces', t1)
        eqs = Equations([eq])

        epbcs, functions = self._get_periodicBCs(domain)
        ebcs = self._get_displacementBCs(domain)
        lcbcs = self._get_linear_combinationBCs(domain)

        ls = ScipyDirect({})

        pb = Problem('elasticity', equations=eqs, auto_solvers=None)

        pb.time_update(
            ebcs=ebcs, epbcs=epbcs, lcbcs=lcbcs, functions=functions)

        ev = pb.get_evaluator()
        nls = Newton({}, lin_solver=ls,
                     fun=ev.eval_residual, fun_grad=ev.eval_tangent_matrix)

        try:
            pb.set_solvers_instances(ls, nls)
        except AttributeError:
            pb.set_solver(nls)

        vec = pb.solve()

        u = vec.create_output_dict()['u'].data
        u_reshape = np.reshape(u, (tuple(x + 1 for x in shape) + u.shape[-1:]))

        dims = domain.get_mesh_bounding_box().shape[1]
        strain = np.squeeze(
            pb.evaluate(
                'ev_cauchy_strain.{dim}.region_all(u)'.format(
                    dim=dims),
                mode='el_avg',
                copy_materials=False))
        strain_reshape = np.reshape(strain, (shape + strain.shape[-1:]))

        stress = np.squeeze(
            pb.evaluate(
                'ev_cauchy_stress.{dim}.region_all(m.D, u)'.format(
                    dim=dims),
                mode='el_avg',
                copy_materials=False))
        stress_reshape = np.reshape(stress, (shape + stress.shape[-1:]))

        return strain_reshape, u_reshape, stress_reshape

    def _get_periodicBCs(self, domain):
        dims = domain.get_mesh_bounding_box().shape[1]

        bc_list_YZ, func_list_YZ = list(
            zip(*[self._get_periodicBC_YZ(domain, i) for i in range(0, dims)]))
        bc_list_X, func_list_X = list(
            zip(*[self._get_periodicBC_X(domain, i) for i in range(1, dims)]))
        return Conditions(
            bc_list_YZ + bc_list_X), Functions(func_list_YZ + func_list_X)
