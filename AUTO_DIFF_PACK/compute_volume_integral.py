import numpy as np
import os
from AUTO_DIFF_PACK import read_h5file_util as rh5


def compute_vol_integral(integrand, i_vec, j_vec, k_vec):
    integral = np.trapz(np.trapz(np.trapz(integrand, k_vec, axis=2), j_vec, axis=1), i_vec, axis=0)
    return integral

def compute_vol_integral_of_field(file_path, file_name, phasename, grid_name, blk_id, field, thresh_flag):
    
    metric = rh5.read_metrics(file_path, file_name, grid_name, blk_id)
    
    integral = np.zeros(metric.xe.shape, dtype=np.float64)
    
    J1 = metric.xe*((metric.yn*metric.zl) - (metric.yl*metric.zl))
    J2 = metric.ye*((metric.xn*metric.zl) - (metric.xl*metric.zn))
    J3 = metric.ze*((metric.xn*metric.yl) - (metric.yn*metric.xl))

    det_J = J1 - J2 + J3
    del J1 
    del J2 
    del J3

    NI, NJ, NK = metric.xe.shape
    i_vec = np.linspace(0,NI-1,NI)
    j_vec = np.linspace(0,NJ-1,NJ)
    k_vec = np.linspace(0,NK-1,NK)

    if (thresh_flag == 1):
        field[field < 0] = 0
        
    integrand = field * det_J
    integral = compute_vol_integral(integrand, i_vec, j_vec, k_vec)
    del integrand
    return integral


    
