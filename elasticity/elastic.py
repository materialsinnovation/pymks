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


def fixed_region_(coors, domain=None):
    eps = 1e-8
    x, y = coors[:, 0], coors[:, 1]
    loc_x = -0.3
    loc_y = 0.0
    y_flag = (y > (loc_y - eps)) & (y < (loc_y + eps))
    x_flag = (x > (loc_x - eps)) & (x < (loc_x + eps))
    flag = nm.where(x_flag & y_flag)[0]
    print flag
    
    return flag

def displaced_region_(coors, domain=None):
    eps = 1e-8
    x, y = coors[:, 0], coors[:, 1]
    loc_x = 0.3
    loc_y = 0.0
    y_flag = (y > (loc_y - eps)) & (y < (loc_y + eps))
    x_flag = (x > (loc_x - eps)) & (x < (loc_x + eps))
    flag = nm.where(x_flag & y_flag)[0]
    print flag
    return flag

def upper_region_(coors, domain=None):
    eps = 1e-8
    x, y = coors[:, 0], coors[:, 1]
    y_flag = (y > (0.5 - eps))
    flag = nm.where(y_flag)[0]
    return flag

def lower_region_(coors, domain=None):
    eps = 1e-8
    x, y = coors[:, 0], coors[:, 1]
    y_flag = (y < (-0.5 + eps))
    flag = nm.where(y_flag)[0]
    return flag

def left_region_(coors, domain=None):
    eps = 1e-8
    x, y = coors[:, 0], coors[:, 1]
    x_flag = (x < (-0.5 + eps))
    flag = nm.where(x_flag)[0]
    return flag

def right_region_(coors, domain=None):
    eps = 1e-8
    x, y = coors[:, 0], coors[:, 1]
    x_flag = (x > (0.5 - eps))
    flag = nm.where(x_flag)[0]
    return flag

usage = """%prog [options]"""
help = {
    'show' : 'show the results figure',
}

def main():
    from sfepy import data_dir
    # import pdb; pdb.set_trace()
    
    parser = OptionParser(usage=usage, version='%prog')
    parser.add_option('-s', '--show',
                      action="store_true", dest='show',
                      default=False, help=help['show'])
    options, args = parser.parse_args()

    mesh = Mesh.from_file(data_dir + '/sfepy/meshes/2d/square_quad.mesh')
    domain = Domain('domain', mesh)

    # min_x, max_x = domain.get_mesh_bounding_box()[:,0]
    # eps = 1e-8 * (max_x - min_x)

    fixed_region = Function('fixed_region',  fixed_region_)
    displaced_region = Function('displaced_region',  displaced_region_)
    upper_region = Function('upper_region',  upper_region_)
    lower_region = Function('lower_region',  lower_region_)
    left_region = Function('left_region',  left_region_)
    right_region = Function('right_region',  right_region_)
    
    domain_all = domain.create_region('Domain_all', 'all')
    domain_fixed = domain.create_region('Domain_fixed', 'vertices by fixed_region', 'vertex', functions=Functions([fixed_region]))
    domain_displaced = domain.create_region('Domain_displaced', 'vertices by displaced_region', 'vertex', functions=Functions([displaced_region]))
    domain_upper = domain.create_region('Domain_upper', 'vertices by upper_region', 'facet', functions=Functions([upper_region]))
    domain_lower = domain.create_region('Domain_lower', 'vertices by lower_region', 'facet', functions=Functions([lower_region]))
    domain_left = domain.create_region('Domain_left', 'vertices by left_region', 'facet', functions=Functions([left_region]))                                              
    domain_right = domain.create_region('Domain_right', 'vertices by right_region', 'facet', functions=Functions([right_region]))
                                             
    
    field = Field.from_args('fu', nm.float64, 'vector', domain_all, approx_order=2)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    E1 = 1.
    E2 = 100.
    v1 = 0.3
    v2 = 0.3

    def _lam(E, v):
        return E * v / (1 + v) / (1 - 2 * v)

    def _mu(E, v):
        K = E / 3 / (1 - 2 * v)
        lam = _lam(E, v)
        return K - lam

    lam1 = _lam(E1, v1)
    mu1 = _mu(E1, v1)
    lam2 = _lam(E2, v2)
    mu2 = _mu(E2, v2)

    print 'lams',lam1,lam2
    print 'mus',mu1,mu2
    
    def lam_func_(ts, coors, mode=None, **kwargs):
        print 'mode',mode
        if mode != 'qp':
            return
        else:
            x, y = coors[:, 0], coors[:, 1]
            mask = (y < 0)
            lam = nm.ones_like(x) * lam1
            lam[mask] = lam2
            mu = nm.ones_like(x) * mu1
            mu[mask] = mu2
            lam.shape = (coors.shape[0], 1, 1)
            mu.shape = lam.shape
            return {'lam' : lam, 'mu' : mu}

    lam_func = Function('lam_func', lam_func_)
    
    m = Material('m', function=lam_func)
    f = Material('f', val=[[0.0], [0.0]])

    integral = Integral('i', order=3)
    
    t1 = Term.new('dw_lin_elastic_iso(m.lam, m.mu, v, u)',
                  integral, domain_all, m=m, v=v, u=u)
    t2 = Term.new('dw_volume_lvf(f.val, v)', integral, domain_all, f=f, v=v)
    eq = Equation('balance', t1 + t2)
    eqs = Equations([eq])

    match_x_line = Function('match_x_line', per.match_x_line)
    periodic_y = PeriodicBC('periodic_y', [domain_upper, domain_lower], {'u.all' : 'u.all'}, match='match_x_line')

    match_y_line = Function('match_y_line', per.match_y_line)
    periodic_x = PeriodicBC('periodic_x', [domain_right, domain_left], {'u.all' : 'u.all'}, match='match_y_line')

    displaced_BC = EssentialBC('displaced_BC', domain_displaced, {'u.0' : 0.1, 'u.1' : 0.0})
    fixed_BC = EssentialBC('fixed_BC', domain_fixed, {'u.0' : 0.0, 'u.1' : 0.0})
    
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
                           right_region])

    pb.time_update(ebcs=Conditions([displaced_BC, fixed_BC]),
                   epbcs=Conditions([periodic_y, periodic_x]),
                   functions=functions)

    vec = pb.solve()
    
    # print nls_status

    pb.save_state('linear_elasticity1.vtk', vec)

    if options.show:
        view = Viewer('linear_elasticity1.vtk')
        view(vector_mode='warp_norm', rel_scaling=2,
             is_scalar_bar=True, is_wireframe=True)

if __name__ == '__main__':
    main()
