
import numpy as np
import sigpy as sp

def rot_matrix(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        mat = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 'y':
        mat = [[c, 0, -s], [0, 1, 0], [s, 0, c]]
    elif axis == 'z':
        mat = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return np.array(mat)

def kspace_rigid_moco(ksp_mov_ctgr, crds_mov_ctgr, mo_estim_ngp, N, motion_frequency=None, device=None):
    """
    Rigid motion correction applied as phase ramp + coordinate rotation in k-space. 
    If device is None, Operations are performed on the same device where the provided k-space data resides.
    
    If <motion_frequency> is not specified 
    Args:
        ksp_mov_ctgr : ndarray, dims - [coil, time, group, repetition]
            k-space data for the case with integroup motion
        crds_mov_ctgr : ndarray, dims - [coordinate, time, group, repetition]
            sampling trajectories for the data with motion
            values are in range [-0.5/N, 0.5/N], where N is image dimension and cubic volume are assumed.
        mo_estim_ngp : ndarray, dims - [poses per group, groups,  parameters]
            estimated 6 rigid motion parameters in pixels/rads
        motion_frequency : int
            frequency [in repetitions] of the motion estimates provided 
            Default: None, motion params are applied every group.
        N : int
            image dimension in pixels
        
    Returns:
        ksp_moco_ctgr : ndarray, dims - [coil, time, group, repetition]
            translationally-corrected k-space data
        crds_rot_ctgr : ndarray, dims - [coordinate, time, group, repetition]
            rotated k-space trajectories
        Returned arrays are on the device set by the argument.
    """
    if device is None:
        device = sp.get_device(ksp_mov_ctgr)
    mvd = lambda x : sp.to_device(x, device)
    mvc = lambda x : sp.to_device(x, sp.Device(-1))
    xp = device.xp
    
    nc, nt, ng, nr = crds_mov_ctgr.shape
    Nt_navi = mo_estim_ngp.shape[0]
    nr_per_motion = nr // Nt_navi #motion_frequency if motion_frequency else nr # Number of TR per motion states 
         
    with device:
        ksp_moco_ctgr_dev = mvd(ksp_mov_ctgr)
        crds_mov_ctgr_dev = mvd(crds_mov_ctgr)
        crds_rot_ctgr = xp.zeros_like(crds_mov_ctgr_dev)

        for gr_ii in range(ng):
            for navi_ii in range(Nt_navi):
                r_begin = nr_per_motion * (navi_ii)
                r_end = nr_per_motion * (navi_ii + 1) #if navi_ii < Nt_navi - 1 else nr
                dx, dy, dz, phi_x, phi_y, phi_z = mo_estim_ngp[navi_ii, gr_ii,:]

                # adjustments from the rotation implementation
                phi_x = -phi_x
                phi_y = -phi_y
                dx = -dx

                Rx = mvd(rot_matrix('x', phi_x))
                Ry = mvd(rot_matrix('y', phi_y))
                Rz = mvd(rot_matrix('z', phi_z))

                crds_c_tr_rot = xp.matmul(Rz, crds_mov_ctgr_dev[:, :, gr_ii, r_begin:r_end].reshape((3, -1)))
                crds_c_tr_rot = xp.matmul(Ry, crds_c_tr_rot)
                crds_c_tr_rot = xp.matmul(Rx, crds_c_tr_rot)

                crds_ctr_rot = crds_c_tr_rot.reshape((3, nt, r_end-r_begin))
                crds_rot_ctgr[:, :, gr_ii, r_begin:r_end] = crds_ctr_rot

                ph_ramp_x_ctr = xp.exp( -2j * xp.pi * dx * crds_ctr_rot[[0]] / N)
                ph_ramp_y_ctr = xp.exp( -2j * xp.pi * dy * crds_ctr_rot[[1]] / N)
                ph_ramp_z_ctr = xp.exp( -2j * xp.pi * dz * crds_ctr_rot[[2]] / N)

                ksp_moco_ctgr_dev[:, :, gr_ii, r_begin:r_end] *= ph_ramp_x_ctr * ph_ramp_y_ctr * ph_ramp_z_ctr
    return ksp_moco_ctgr_dev, crds_rot_ctgr
