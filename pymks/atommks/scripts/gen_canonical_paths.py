import os
import sys
sys.path.append("../../../")

import os
import ase
import time
import glob
import numpy as np
import pandas as pd
import ase.io as aio
import scipy.io as sio
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from toolz.curried import pipe, curry, compose
from collections import defaultdict, OrderedDict

import pymks.atommks.porosity as pore
from pymks.atommks.helpers import write2vtk
from pymks.atommks.helpers import save_file, load_file
from pymks.atommks.grid_generator import generate_grids

from pymks.atommks.canonical_paths import calc_path_distance, calc_path_distances_matrix, calc_canonical_paths


def get_radius(atom_id, radius_type="vdw"):
    """
    Get the radius of the atom
    
    Args:
      atom_id: element symbol
      radius_type = "vdw" for Van der Waals or "cov" for Covalent
      
    Returns:
      the atomic radius
      
    >>> get_radius('Na')
    2.27
    """
    xl = pd.ExcelFile("Elemental_Radii.xlsx")
    df = xl.parse(sheet_name=0, header = 2, index_col=1)
    
    if radius_type is "cov":
        key = 6
    elif radius_type is "vdw":
        key = 7
    else:
        raise ValueError("radius_type not supported")
    if atom_id in df.index:
        return df.loc[atom_id][key]
    else:
        raise ValueError("Elemental symbol not found")


def get_structure_data(cif_file_path, resize_unit_cell=1):
    """
    Get the ASE atom object (a molecule in many cases) and corresponding
    radii for each atom in the molecule
    
    Args:
      cif_file_path: path to the CIF file
      resize_unit_cell: allows a resize of the atom object
      
    Returns:
      a tuple of the ASE atom object and dictionary of atom radii
    
    >>> get_structure_data('iza_zeolites/DDR.cif')[0].get_cell_lengths_and_angles()
    array([ 27.59,  27.59,  81.5 ,  90.  ,  90.  , 120.  ])
    
    """
    ase_atom = aio.read(cif_file_path).repeat(resize_unit_cell if hasattr(resize_unit_cell, "__len__") else [resize_unit_cell] * 3)
    atom_ids = sorted(np.unique(ase_atom.get_chemical_symbols()))
    return (
        ase_atom,
        {idx:get_radius(idx) for idx in atom_ids}
    )

if __name__=="__main__":
        
    file_list = glob.glob("../../../../pore-analytics/structures/likely-min-energy-structures/*.cif")
    
    print(f"No. of structures: {len(file_list)}")
    
    for i0, file_path in enumerate(file_list):

        cif = file_path.split("/")[-1][:-4]

        print(i0+1, cif)

        ase_atom, radii = get_structure_data(Path(file_path), [2, 2, 1])

        grid_data = generate_grids(
            ase_atom,
            n_pixel=10,
            atomic_radii=radii,
            extend_boundary_atoms=False,
            use_fft_method=False
        )
        
        grid_data["distance_grid"], metrics = pore.calc_pore_metrics(grid_data['pores'], n_pixel=grid_data['n_pixel'])


        dists_dict, canonical_dists_dict = pore.calc_diffusion_paths(grid_data["distance_grid"], 
                                                                     r_probe=0.5, 
                                                                     n_pixel=grid_data["n_pixel"])  
        
        print([(k, len(v)) for (k, v) in canonical_dists_dict.items()])

        save_file(obj=canonical_dists_dict, fname=f"likely_min_canonicals/canonical_paths_dict_{cif}.pkl")