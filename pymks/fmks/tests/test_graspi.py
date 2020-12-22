import pymks.fmks.graspi as graspi
import numpy as np

def test_compute_descriptors():
    morph = np.array( [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0 ],dtype=np.int32)
    desc = graspi.compute_descriptors(morph, 4, 3, 1, 2, 1)
    result = [(12.0, b'STAT_n'), (8.0, b'STAT_e'), (6.0, b'STAT_n_D'), (6.0, b'STAT_n_A'), (1.0, b'STAT_CC_D'), (1.0, b'STAT_CC_A'), (1.0, b'STAT_CC_D_An'), (1.0, b'STAT_CC_A_Ca'), (0.4804587662220001, b'ABS_wf_D'), (0.5, b'ABS_f_D'), (0.8879592418670654, b'DISS_wf10_D'), (1.0, b'DISS_f10_D'), (1.0, b'CT_f_e_conn'), (1.0, b'CT_f_conn_D_An'), (1.0, b'CT_f_conn_A_Ca'), (8.0, b'CT_e_conn'), (6.0, b'CT_e_D_An'), (6.0, b'CT_e_A_Ca'), (1.0, b'CT_f_D_tort1'), (0.8333333134651184, b'CT_f_A_tort1')]
    assert desc == result
