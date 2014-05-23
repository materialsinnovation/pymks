import tempfile
from subprocess import Popen, PIPE
import os


import numpy as np
from sfepy.base.base import IndexedStruct
from sfepy.discrete.fem import Mesh, Domain, Field
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess import Viewer
import sfepy.discrete.fem.periodic as per
from sfepy.discrete import Functions


class ElasticFEModel(object):
    def __init__(self, N, L=1.):
        if N % 2 == 0:
            raise ValueError, "N must be odd"
        geo_string = self.createGEOstring(N, L=L)
        self.createMSHfile(geo_string)
        
    def createGEOstring(self, N, L=1.):
        dx = L / N
        self.dx = dx
        # kludge: must offset cellSize by `eps` to work properly
        eps = float(dx) / (N * 10) 

        template_file = 'template.geo'
        template_path = os.path.split(__file__)[0]
        with open(os.path.join(template_path, template_file)) as f:
            template = f.read()

        return template.format(N=N, dx=dx, L=L, eps=eps)
            
    def createMSHfile(self, gmsh_string):
        r"""
        Write a Gmsh MSH file from a .geo string

        Args:
          gmsh_string: the geo file string

        Returns:
          The name of the MSH file
        """
        dimensions = 2
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geo', delete=False) as f_geo:
            f_geo.writelines(gmsh_string)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.msh', delete=False) as f_msh:
            pass

        gmshFlags = ["-%d" % dimensions, "-nopopup", "-format", "msh"]
        p = Popen(["gmsh", f_geo.name] + gmshFlags + ["-o", f_msh.name], stdout=PIPE)
        gmshOutput, gmshError = p.communicate()
        gmshOutput = gmshOutput.decode('ascii')
        print(gmshOutput)

        os.remove(f_geo.name)
        self.msh_file = f_msh.name

    def convert_properties(self, elastic_modulus, poissons_ratio):
        E = elastic_modulus
        nu = poissons_ratio
        lame = E * nu / (1 + nu) / (1 - 2 * nu)
        K = E / 3 / (1 - 2 * nu)
        shear = K - lame
        return lame, shear
                
    def predict(self, strain, elastic_modulus, poissons_ratio):
        if (len(elastic_modulus) != 2) or (len(poissons_ratio) == 2):
            raise ValueError, 'elastic_modulus and poissons_ration must be length 2'
        lame, shear_modulus = self.convert_properties(elastic_modulus, poissons_ratio)

    def get_material(self, lame, shear_modulus, domain):
        bbox = domain.get_mesh_bounding_box()
        x0, y0 = (bbox[1,:] + bbox[0,:]) / 2.
        eps = 1e-8 * (bbox[1,0] - bbox[0,0])
        lx = self.dx / 2 + eps

        def material_func_(ts, coors, mode=None, **kwargs):
            if mode != 'qp':
                return
            else:
                x, y = coors[:, 0], coors[:, 1]
                mask_x = (x < (x0 + lx)) & (x > (x0 - lx))
                mask_y = (y < (y0 + lx)) & (y > (y0 - lx))
                mask = mask_x & mask_y
                if np.sum(mask) != 4:
                    raise RuntimeError, "mask should be length 4"
                lam = np.ones_like(x) * lame[0]
                lam[mask] = lame[1]
                mu = np.ones_like(x) * shear_modulus[0]
                my[mask] = shear_modulus[1]
                return {'lam' : lam, 'mu' : mu}

        material_func = Function('material_func', material_func_)
        return Material('m', function=material_func)

    def solve(self, strain, lame, shear_modulus):
        mesh = Mesh.from_file(self.msh_file)
        domain = Domain('domain', mesh)

        region_all = domain.create_region('region_all', 'all')
        
        field = Field.from_args('fu', np.float64, 'vector', region_all, approx_order=2)

        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        m = self.get_material(lame, shear_modulus, domain)
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

        functions = Functions([match_x_line,
                               match_y_line,
                               displaced_region,
                               fixed_region,
                               upper_region,
                               lower_region,
                               left_region,
                               right_region,
                               lam_func])

        pb.time_update(ebcs=Conditions([displaced_BC, fixed_BC]),
        epbcs=Conditions([periodic_y, periodic_x]),
        functions=functions)

        vec = pb.solve()

        return vec

        
    def __del__(self):
        os.remove(self.msh_file)

    
