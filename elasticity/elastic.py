#!/usr/bin/env python
from optparse import OptionParser
import numpy as nm

import sys
sys.path.append('.')



from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, Domain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess import Viewer
import sfepy.discrete.fem.periodic as per
from sfepy.discrete import Functions


def shift_u_fun(ts, coors, bc=None, problem=None, shift=0.0):
    """
    Define a displacement depending on the y coordinate.
    """
    val = shift * nm.ones_like(coors[:, 1])
    return val

def inner_region_(coors, domain=None):
    x, y = coors[:, 0], coors[:, 1]
    flag = nm.where((x < 0.2) & (x > -0.2) & (y > -0.2) & (y < 0.2))[0]
    return flag

usage = """%prog [options]"""
help = {
    'show' : 'show the results figure',
}

def main():
    from sfepy import data_dir
    import pdb; pdb.set_trace()
    
    parser = OptionParser(usage=usage, version='%prog')
    parser.add_option('-s', '--show',
                      action="store_true", dest='show',
                      default=False, help=help['show'])
    options, args = parser.parse_args()

    mesh = Mesh.from_file(data_dir + '/sfepy/meshes/2d/square_quad.mesh')
    domain = Domain('domain', mesh)

    min_x, max_x = domain.get_mesh_bounding_box()[:,0]
    eps = 1e-8 * (max_x - min_x)
    min_y, max_y = domain.get_mesh_bounding_box()[:,1]

    inner_region = Function('inner_region',  inner_region_)
    
    omega = domain.create_region('Omega', 'all')
    omega1 = domain.create_region('Omega1', 'vertices by inner_region', functions=Functions([inner_region]))
    gamma1 = domain.create_region('Gamma1',
                                  'vertices in x < %.10f' % (min_x + eps),
                                  'facet')
    gamma2 = domain.create_region('Gamma2',
                                  'vertices in x > %.10f' % (max_x - eps),
                                  'facet')
    gamma_bottom = domain.create_region('Gamma_bottom',
                                        'vertices in y < %.10f' % (min_y + eps),
                                        'facet')
    gamma_top = domain.create_region('Gamma_top',
                                     'vertices in y > %.10f' % (max_y - eps),
                                     'facet')


    
    field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order=2)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    def lam_func_(ts, coors, mode=None, **kwargs):
        if mode != 'qp':
            return
        else:
            value = 1. * (coors[:, 0] > .25) + 1.
            value.shape = (coors.shape[0], 1, 1)
            
            one = nm.ones_like(value)
            return {'lam' : value, 'mu' : one}

    lam_func = Function('lam_func_', lam_func_)
        
    m = Material('m', function=lam_func)
    f = Material('f', val=[[0.0], [0.0]])

    integral = Integral('i', order=3)

    t1 = Term.new('dw_lin_elastic_iso(m.lam, m.mu, v, u)',
                  integral, omega, m=m, v=v, u=u)
    t2 = Term.new('dw_volume_lvf(f.val, v)', integral, omega, f=f, v=v)
    eq = Equation('balance', t1 + t2)
    eqs = Equations([eq])

    #fix_u = EssentialBC('fix_u', gamma1, {'u.0' : 0.0})

    # bc_fun = Function('shift_u_fun', shift_u_fun, extra_args={'shift' : 0.0})
    # shift_u = EssentialBC('shift_u', gamma2, {'u.0' : bc_fun})

    # bc_fun = Function('shift_u_fun', shift_u_fun, extra_args={'shift' : 0.0})
    # shift_u = EssentialBC('shift_u', gamma2, {'u.0' : bc_fun})
    
    match_x_line = Function('match_x_line', per.match_x_line)
    periodic_y = PeriodicBC('periodic_y', [gamma_top, gamma_bottom], {'u.all' : 'u.all'}, match='match_x_line')

    match_y_line = Function('match_y_line', per.match_y_line)
    periodic_x = PeriodicBC('periodic_x', [gamma1, gamma2], {'u.all' : 'u.all'}, match='match_y_line')

    inner_displacement = EssentialBC('inner_displacement', omega1, {'u.0' : 0.01, 'u.1' : 0.0})
    
    ls = ScipyDirect({})

    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver=ls, status=nls_status)

    pb = Problem('elasticity', equations=eqs, nls=nls, ls=ls)
    pb.save_regions_as_groups('regions')
    # functions = Functions([match_x_line, match_y_line])
    # pb.time_update(ebcs=Conditions([fix_u]),
    #                epbcs=Conditions([periodic_y, periodic_x]),
    #                functions=functions)
    functions = Functions([match_x_line, match_y_line, inner_region])
    pb.time_update(ebcs=Conditions([inner_displacement]),
                   epbcs=Conditions([periodic_y, periodic_x]),
                   functions=functions)

    vec = pb.solve()
    print nls_status

    pb.save_state('linear_elasticity.vtk', vec)

    if options.show:
        view = Viewer('linear_elasticity.vtk')
        view(vector_mode='warp_norm', rel_scaling=2,
             is_scalar_bar=True, is_wireframe=True)

if __name__ == '__main__':
    main()
