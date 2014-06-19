import numpy as np
from sfepy.discrete import Functions
from pymks import ElasticFEModel
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
import sfepy.discrete.fem.periodic as per
try:
    from sfepy.discrete.fem import FEDomain as Domain
except ImportError:
    from sfepy.discrete.fem import Domain
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term
from sfepy.discrete.fem import Field
from sfepy.base.base import IndexedStruct

class ElasticFE3DModel(ElasticFEModel):
    """

    Use SfePy to solve a linear strain problem in 3D cube with a
    varying microstructure on a rectangular grid. The cube is held at
    the x_minus plain and the x_plus plain is displaced by the length
    of the x dimension times the macroscopic stain. Periodic boundary
    conditions are applied to the upper and lower plains.

    The microstructure is of shape (Nsample, Nx, Ny, Nproperty) where
    Nproperty is 2 for the elastic modulus and Poisson's ratio.

    >>> X = np.ones((1, 3, 3, 3, 2))

    >>> model = ElasticFE3DModel(dx=0.2)
    >>> y = model.predict(X) # doctest: +ELLIPSIS
    sfepy: ...
    
    y is the strain with components as follows
    >>> print 'y0',y[..., 0]
    >>> print 'y1',y[..., 1]
    >>> print 'y2',y[..., 2]
    >>> print 'y3',y[..., 3]
    >>> print 'y4',y[..., 4]
    >>> print 'y5',y[..., 5]
    >>> exx = y[..., 0]
    >>> eyy = y[..., 1]
    >>> exy = y[..., 2]

    Since there is no contrast in the microstructure the strain is only
    in the x-direction and has a uniform value of 1 since the
    displacement is always 1 and the size of the domain is 1.

    >>> #assert np.allclose(exx, 1)
    >>> #assert np.allclose(eyy, 0)
    >>> #assert np.allclose(exy, 0)

    """

    def predict(self, X):
        """
        Predict the displacement field given an initial microstructure
        and a strain in the x direction.

        Args:
          X: microstructure with shape (Nsample, Nx, Ny, Nz, Nproperty)
             with len(Nproperty) = 2. X[..., 0] represents the elastic
             modulus and X[..., 1] is the Poisson's ratio

        Returns:
          the strain field over each cell
          
        """
        Nsample, Nx, Ny, Nz, Nproperty = X.shape
        if (Nproperty != 2):
            raise RuntimeError, 'the shape of X is incorrect'
        
        X_ = self.convert_properties(X)

        y_strain = np.array([self.solve(x) for x in X_])

        return y_strain


    def get_material(self, property_array, domain):
        """
        Creates an SfePy material from the material property fields

        Args:
          property_array: array of the properties with shape (Nx, Ny, 2)

        Returns:
          an SfePy material
          
        """
        minx, maxx = domain.get_mesh_bounding_box()[:, 0]
        miny, maxy = domain.get_mesh_bounding_box()[:, 1]
        minz, maxz = domain.get_mesh_bounding_box()[:, 2]

        def material_func_(ts, coors, mode=None, **kwargs):
            if mode == 'qp':
                x, y, z = coors[:, 0], coors[:, 1], coors[:, 2]
                i_out = np.empty_like(x, dtype=np.int64)
                j_out = np.empty_like(y, dtype=np.int64)
                k_out = np.empty_like(z, dtype=np.int64)
                i = np.floor((x - minx) / self.dx, i_out)
                j = np.floor((y - miny) / self.dx, j_out)
                k = np.floor((z - minz) / self.dx, k_out)
                property_array_ = property_array[i, j, k]
                lam = property_array_[..., 0]
                mu = property_array_[..., 1]
                lam = np.ascontiguousarray(lam.reshape((lam.shape[0], 1, 1)))
                mu = np.ascontiguousarray(mu.reshape((mu.shape[0], 1, 1)))
                return {'lam' : lam, 'mu' : mu}
            else:
                return

        material_func = Function('material_func', material_func_)
        return Material('m', function=material_func)

    def get_periodicBCs(self, domain):
        """
        Creates periodic boundary conditions with the top and bottom y-planes.

        Args:
          domain: an Sfepy domain

        Returns:
          a tuple of Sfepy boundary condition and associated matching functions

        """
        miny, maxy = domain.get_mesh_bounding_box()[:, 1]
        minz, maxz = domain.get_mesh_bounding_box()[:, 2]
        yplus_ = self.subdomain_func(y=(maxy,))
        yminus_ = self.subdomain_func(y=(miny,))
        yplus = Function('yplus', yplus_)
        yminus = Function('yminus', yminus_)
        zplus_ = self.subdomain_func(z=(maxz,))
        zminus_ = self.subdomain_func(z=(minz,))
        zplus = Function('zplus', zplus_)
        zminus = Function('zminus', zminus_)
        region_y_plus = domain.create_region('region_y_plus',
                                         'vertices by yplus',
                                         'facet',
                                             functions=Functions([yplus]))
        region_y_minus = domain.create_region('region_y_minus',
                                           'vertices by yminus',
                                           'facet',
                                              functions=Functions([yminus]))

        region_z_plus = domain.create_region('region_z_plus',
                                         'vertices by zplus',
                                         'facet',
                                             functions=Functions([zplus]))
        region_z_minus = domain.create_region('region_z_minus',
                                           'vertices by zminus',
                                           'facet',
                                              functions=Functions([zminus]))
        match_y_plane = Function('match_y_plane', per.match_y_plane)
        match_z_plane = Function('match_z_plane', per.match_z_plane)
        periodic_y = PeriodicBC('periodic_y', [region_y_plus, region_y_minus], {'u.all' : 'u.all'}, match='match_y_plane')
        periodic_z = PeriodicBC('periodic_z', [region_z_plus, region_z_minus], {'u.all' : 'u.all'}, match='match_z_plane')
        return Conditions([periodic_y, periodic_z]), Functions([match_y_plane, match_z_plane])


    def get_displacementBCs(self, domain):
        """Fix the x_minus plane, displace the x_plus plane by the length of
        the dimenision x times macroscopic strain and fix the
        y-direction with the top and bottom points on the left x
        plane.

        Args:
          domain: an Sfepy domain

        Returns:
          the Sfepy boundary conditions

        """
        minx, maxx = domain.get_mesh_bounding_box()[:, 0]
        miny, maxy = domain.get_mesh_bounding_box()[:, 1]
        minz, maxz = domain.get_mesh_bounding_box()[:, 2]

        xplus_ = self.subdomain_func(x=(maxx,))
        
        xminus_ = self.subdomain_func(x=(minx,))
        fix_x_points_ = self.subdomain_func(x=(minx,), y=(maxy, miny), z=(maxz, minz))
        xplus = Function('xplus', xplus_)
        xminus = Function('xminus', xminus_)
        fix_x_points = Function('fix_x_points', fix_x_points_)
        region_x_plus = domain.create_region('region_x_plus',
                                            'vertices by xplus',
                                            'facet',
                                            functions=Functions([xplus]))
        region_x_minus = domain.create_region('region_x_minus',
                                           'vertices by xminus',
                                           'facet',
                                           functions=Functions([xminus]))
        region_fix_points = domain.create_region('region_fix_points',
                                          'vertices by fix_x_points',
                                          'vertex',
                                          functions=Functions([fix_x_points]))
        fixed_BC = EssentialBC('fixed_BC', region_x_minus, {'u.all' : 0.0})
        displaced_BC = EssentialBC('displaced_BC', region_x_plus, {'u.all' : 0 * self.strain * (maxx - minx)})
        fixed_points_BC = EssentialBC('fix_points_BC', region_fix_points, {'u.all' : 0.0})
        
        #return Conditions([fixed_BC, displaced_BC, fixed_points_BC])
        
        miny, maxy = domain.get_mesh_bounding_box()[:, 1]
        minz, maxz = domain.get_mesh_bounding_box()[:, 2]
        yplus_ = self.subdomain_func(y=(maxy,))
        yminus_ = self.subdomain_func(y=(miny,))
        yplus = Function('yplus', yplus_)
        yminus = Function('yminus', yminus_)
        zplus_ = self.subdomain_func(z=(maxz,))
        zminus_ = self.subdomain_func(z=(minz,))
        zplus = Function('zplus', zplus_)
        zminus = Function('zminus', zminus_)
        region_y_plus = domain.create_region('region_y_plus',
                                         'vertices by yplus',
                                         'facet',
                                             functions=Functions([yplus]))
        region_y_minus = domain.create_region('region_y_minus',
                                           'vertices by yminus',
                                           'facet',
                                              functions=Functions([yminus]))

        region_z_plus = domain.create_region('region_z_plus',
                                         'vertices by zplus',
                                         'facet',
                                             functions=Functions([zplus]))
        region_z_minus = domain.create_region('region_z_minus',
                                           'vertices by zminus',
                                           'facet',
                                              functions=Functions([zminus]))
        

        
        fixed_y_minus_BC = EssentialBC('fixed_y_minus_BC', region_y_minus, {'u.all' : 0.0})
        fixed_y_plus_BC = EssentialBC('fixed_y_plus_BC', region_y_plus, {'u.all' : 0.0})
        fixed_z_minus_BC = EssentialBC('fixed_z_minus_BC', region_z_minus, {'u.all' : 0.0})
        fixed_z_plus_BC = EssentialBC('fixed_z_plus_BC', region_z_plus, {'u.all' : 0.0})

        return Conditions([fixed_BC, displaced_BC, fixed_points_BC, fixed_y_minus_BC, fixed_y_plus_BC, fixed_z_minus_BC, fixed_z_plus_BC ])
        
    def get_mesh(self, shape):
        """
        Generate an Sfepy rectangular mesh

        Args:
          shape: proposed shape of domain (vertex shape) (Nx, Ny, Nz)

        Returns:
          Sfepy mesh
          
        """
        Lx = shape[0] * self.dx
        Ly = shape[1] * self.dx
        Lz = shape[2] * self.dx
        center = (0., 0., 0.)
        vertex_shape = (shape[0] + 1, shape[1] + 1, shape[2] + 1)
        return gen_block_mesh((Lx, Ly, Lz), vertex_shape, center, verbose=False)
    
    def solve(self, property_array):
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
        mesh = self.get_mesh(shape)
        domain = Domain('domain', mesh)

        region_all = domain.create_region('region_all', 'all')
        
        field = Field.from_args('fu', np.float64, 'vector', region_all, approx_order=2)

        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        m = self.get_material(property_array, domain)
        f = Material('f', val=[[0.0], [0.0]])

        integral = Integral('i', order=3)
    
        t1 = Term.new('dw_lin_elastic_iso(m.lam, m.mu, v, u)',
                      integral, region_all, m=m, v=v, u=u)
        t2 = Term.new('dw_volume_lvf(f.val, v)', integral, region_all, f=f, v=v)
        eq = Equation('balance', t1 + t2)
        eqs = Equations([eq])

        ls = ScipyDirect({})
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)

        pb = Problem('elasticity', equations=eqs, nls=nls, ls=ls)
        pb.save_regions_as_groups('regions')

        #epbcs, functions = self.get_periodicBCs(domain)
        
        ebcs = self.get_displacementBCs(domain)

        #pb.time_update(ebcs=ebcs,
        #               epbcs=epbcs,
        #               functions=functions)
        
        pb.time_update(ebcs=ebcs)
        vec = pb.solve()
       
        #pb.solve()
        #strain = np.squeeze(pb.evaluate('ev_cauchy_strain.3.region_all(u)', mode='el_avg'))

        #return np.reshape(strain, (shape + strain.shape[-1:]))
        return vec.create_output_dict()['u'].data

