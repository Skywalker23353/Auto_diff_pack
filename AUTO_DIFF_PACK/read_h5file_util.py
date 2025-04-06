import h5py
from dataclasses import dataclass
import numpy as np
import os

def hdf5_read_LES_data(file_path,file_name,findx,phasename,grid_name,blkid,field_name):
        file_name = file_name + '_' + str(findx) + '.h5'
        file_path = os.path.join(file_path,file_name)
        dataset_path =  '/' + phasename + '/' + grid_name + '/' + 'fields' + '/' + str(blkid) + '/' + field_name 
        with h5py.File(file_path, 'r') as f:
            grid_data_buf = np.array(f[dataset_path], dtype=np.float64)

        return (grid_data_buf)

def hdf5_read_grids_file(file_path,file_name,grid_name,blkid,field_name):
        file_name = file_name + '_grids_0.h5'
        file_path = os.path.join(file_path,file_name)
        dataset_path =  '/' + grid_name + '/' + 'source_blocks' + '/' + str(blkid) + '/' + field_name 
        with h5py.File(file_path, 'r') as f:
            grid_data_buf = np.array(f[dataset_path], dtype=np.float64)

        return (grid_data_buf)

def read_metrics(file_path,file_name,grid_name,blkid):
        @dataclass
        class MetricsData:
                xn: np.ndarray
                xe: np.ndarray
                xl: np.ndarray
                yn: np.ndarray
                ye: np.ndarray
                yl: np.ndarray
                zn: np.ndarray
                ze: np.ndarray
                zl: np.ndarray
                nx: np.ndarray
                ny: np.ndarray
                nz: np.ndarray
                ex: np.ndarray
                ey: np.ndarray
                ez: np.ndarray
                lx: np.ndarray
                ly: np.ndarray
                lz: np.ndarray

        xn = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricxn')
        xe = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricxe')
        xl = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricxl')
        yn = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricyn')
        ye = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricye')
        yl = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricyl')
        zn = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metriczn')
        ze = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricze')
        zl = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metriczl')
        nx = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricnx')
        ny = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricny')
        nz = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricnz')
        lx = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metriclx')
        ly = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricly')
        lz = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metriclz')
        ex = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricex')
        ey = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricey')
        ez = hdf5_read_grids_file(file_path, file_name, grid_name, blkid, 'metricez')


        return MetricsData(xn, xe, xl, yn, ye, yl, zn, ze, zl, nx, ny, nz, ex, ey, ez, lx, ly, lz)
        