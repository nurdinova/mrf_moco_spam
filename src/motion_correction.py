
import numpy as np
import sigpy as sp

        
def motion_corr_intergroup(ksp_mov_ctgr, crds_mov_ctgr, mo_estim, N, in_vivo=False, device=None):
    """
    Args:
        ksp_mov_ctgr : ndarray, dims - [coil, time, group, repetition]
            k-space data for the case with integroup motion
        crds_mov_ctgr : ndarray, dims - [coordinate, time, group, repetition]
            spiral sampling trajectories for the data with motion 
        mo_estim : ndarray, dims - [poses, parameters]
            estimated per-group 6 rigid motion parameters
        in_vivo : bool
            flag to indicate if the data is in vivo or simulated
        
    For the in vivo data, motion is estimated with respect to the first group.
    For the simulated data, we have reference 0-motion state, so parameters are estimated for all shots.
    
    Returns:
        ksp_mov_ctgr_moco : ndarray, dims - [coil, time, group, repetition]
            translationally-corrected k-space data
        crds_ctgr_rot : ndarray, dims - [coordinate, time, group, repetition]
            rotated spiral trajectories
    """
    if device is None:
        device = sp.cpu_device
    mvd = lambda x : sp.to_device(x, device)
    mvc = lambda x : sp.to_device(x, sp.Device(-1))
    xp = device.xp
    
    ksp_mov_ctgr_moco_dev = mvd(xp.copy(ksp_mov_ctgr))
    crds_mov_ctgr_dev = mvd(crds_mov_ctgr)
    crds_ctgr_rot_dev = xp.zeros_like(crds_mov_ctgr) # we gonna rotate the spiral trajectories

    _, nt, ng, nr = crds_mov_ctgr.shape

    for gr_ii in range(ng):
        if in_vivo:
            if gr_ii == 0:
                dx, dy, dz, phi_x, phi_y, phi_z = xp.zeros(6,)
            else:
                dx, dy, dz, phi_x, phi_y, phi_z = mo_estim[gr_ii-1,:]
        else:
            dx, dy, dz, phi_x, phi_y, phi_z = mo_estim[gr_ii,:]

        # adjustments according to the rotation implementation
        phi_x = -phi_x
        phi_y = -phi_y
        dx = -dx

        Rx = xp.array([[1, 0, 0], [0, xp.cos(phi_x), -xp.sin(phi_x)], \
              [0, xp.sin(phi_x), xp.cos(phi_x)]])
        Ry = xp.array([[xp.cos(phi_y), 0, -xp.sin(phi_y)], [0, 1, 0], \
              [xp.sin(phi_y), 0, xp.cos(phi_y)]])
        Rz = xp.array([[xp.cos(phi_z), -xp.sin(phi_z), 0], \
                       [xp.sin(phi_z), xp.cos(phi_z), 0], [0, 0, 1]])

        # crds are in the range [-0.5, 0.5]
        crds_c_tr_rot = xp.matmul(Rz, crds_mov_ctgr_dev[:, :, gr_ii, :].reshape((3, -1)))
        crds_c_tr_rot = xp.matmul(Ry, crds_c_tr_rot)
        crds_c_tr_rot = xp.matmul(Rx, crds_c_tr_rot)
        crds_ctr_rot = crds_c_tr_rot.reshape((3, nt, nr))
        crds_ctgr_rot_dev[:, :, gr_ii, :] = crds_ctr_rot

        ph_ramp_x_ctr = xp.tile(xp.exp( -2j * xp.pi * dx * crds_ctr_rot[[0]] / N ), (nc, 1, 1))
        ph_ramp_y_ctr = xp.tile(xp.exp( -2j * xp.pi * dy * crds_ctr_rot[[1]] / N ), (nc, 1, 1))
        ph_ramp_z_ctr = xp.tile(xp.exp( -2j * xp.pi * dz * crds_ctr_rot[[2]] / N ), (nc, 1, 1))

        ksp_mov_ctgr_moco_dev[:, :, gr_ii, :] = ksp_mov_ctgr_moco_dev[:, :, gr_ii, :]  \
            * ph_ramp_x_ctr * ph_ramp_y_ctr * ph_ramp_z_ctr
        
    return mvc(ksp_mov_ctgr_moco), mvc(crds_ctgr_rot)


def motion_corr_interTR(ksp_mov_ctgr, crds_mov_ctgr, mo_estim_ngp, N, in_vivo=False, device=None):
    if device is None:
        device = sp.cpu_device
    mvd = lambda x : sp.to_device(x, device)
    mvc = lambda x : sp.to_device(x, sp.Device(-1))
    xp = device.xp
    
    with device:
        ksp_mov_ctgr_moco_dev = mvd(ksp_mov_ctgr)
        crds_mov_ctgr_dev = mvd(crds_mov_ctgr)
        crds_ctgr_rot_dev = xp.zeros_like(crds_mov_ctgr_dev)

        Nt_navi = 10
        _, nt, N_groups, _ = crds_mov_ctgr.shape
        nr = 47
        for gr_ii in range(N_groups):
            for navi_ii in range(Nt_navi):
                dx, dy, dz, phi_x, phi_y, phi_z = mo_estim_ngp[navi_ii, gr_ii,:]

                # adjustments from the rotation implementation
                phi_x = -phi_x
                phi_y = -phi_y
                dx = -dx

                Rx = mvd(np.array([[1, 0, 0], [0, np.cos(phi_x), -np.sin(phi_x)], \
                      [0, np.sin(phi_x), np.cos(phi_x)]]))
                Ry = mvd(np.array([[np.cos(phi_y), 0, -np.sin(phi_y)], [0, 1, 0], \
                      [np.sin(phi_y), 0, np.cos(phi_y)]]))
                Rz = mvd(np.array([[np.cos(phi_z), -np.sin(phi_z), 0], \
                               [np.sin(phi_z), np.cos(phi_z), 0], [0, 0, 1]]))

                # crds are in [-0.5, 0.5]
                crds_c_tr_rot = xp.matmul(Rz, crds_mov_ctgr_dev[:, :, gr_ii, nr*(navi_ii):nr*(navi_ii+1)].reshape((3, -1)))
                crds_c_tr_rot = xp.matmul(Ry, crds_c_tr_rot)
                crds_c_tr_rot = xp.matmul(Rx, crds_c_tr_rot)

                crds_ctr_rot = crds_c_tr_rot.reshape((3, nt, nr))
                crds_ctgr_rot_dev[:, :, gr_ii, nr*(navi_ii):nr*(navi_ii+1)] = crds_ctr_rot

                ph_ramp_x_ctr = xp.tile(xp.exp( -2j * xp.pi * dx * crds_ctr_rot[[0]] / N ), (nc, 1, 1))
                ph_ramp_y_ctr = xp.tile(xp.exp( -2j * xp.pi * dy * crds_ctr_rot[[1]] / N ), (nc, 1, 1))
                ph_ramp_z_ctr = xp.tile(xp.exp( -2j * xp.pi * dz * crds_ctr_rot[[2]] / N ), (nc, 1, 1))

                ksp_mov_ctgr_moco_dev[:, :, gr_ii, nr*(navi_ii):nr*(navi_ii+1)] *= ph_ramp_x_ctr * ph_ramp_y_ctr * ph_ramp_z_ctr
    return mvc(ksp_mov_ctgr_moco_dev), mvc(crds_ctgr_rot_dev)


# def motion_corr_interTR_centered(ksp_mov_ctgr, crds_mov_ctgr, mo_estim_ngp, in_vivo=False):
#     ksp_mov_ctgr_moco = np.copy(ksp_mov_ctgr)
#     crds_ctgr_rot = np.zeros_like(crds_mov_ctgr)
    
#     Nt_navi = 10
#     _, nt, N_groups, _ = crds_mov_ctgr.shape
#     nr = 47
#     for gr_ii in range(N_groups):
#         for navi_ii in range(Nt_navi):
#             dx, dy, dz, phi_x, phi_y, phi_z = mo_estim_ngp[navi_ii, gr_ii,:]

#             # adjustments from the rotation implementation
#             phi_x = -phi_x
#             phi_y = -phi_y
#             dx = -dx

#             Rx = np.array([[1, 0, 0], [0, np.cos(phi_x), -np.sin(phi_x)], \
#                   [0, np.sin(phi_x), np.cos(phi_x)]])
#             Ry = np.array([[np.cos(phi_y), 0, -np.sin(phi_y)], [0, 1, 0], \
#                   [np.sin(phi_y), 0, np.cos(phi_y)]])
#             Rz = np.array([[np.cos(phi_z), -np.sin(phi_z), 0], \
#                            [np.sin(phi_z), np.cos(phi_z), 0], [0, 0, 1]])

#             # crds are in [-0.5, 0.5]
#             begin = nr*(navi_ii+1)-nr//2
#             tr_corr = np.arange(begin,begin+nr,1)
#             crds_c_tr_rot = Rx @ Ry @ Rz @ crds_mov_ctgr[:,:,gr_ii,tr_corr].reshape((3, -1))
#             crds_ctr_rot = crds_c_tr_rot.reshape((3, nt, nr))
#             crds_ctgr_rot[:,:,gr_ii,tr_corr] = crds_ctr_rot

#             ph_ramp_x_ctr = np.tile(np.exp( -2j * np.pi * dx * crds_ctr_rot[[0]] / N ), (nc, 1, 1))
#             ph_ramp_y_ctr = np.tile(np.exp( -2j * np.pi * dy * crds_ctr_rot[[1]] / N ), (nc, 1, 1))
#             ph_ramp_z_ctr = np.tile(np.exp( -2j * np.pi * dz * crds_ctr_rot[[2]] / N ), (nc, 1, 1))

#             ksp_mov_ctgr_moco[:,:,gr_ii,47*(navi_ii):47*(navi_ii+1)] *= ph_ramp_x_ctr * ph_ramp_y_ctr * ph_ramp_z_ctr
#     return ksp_mov_ctgr_moco, crds_ctgr_rot

def motion_corr_interTR_next(ksp_mov_ctgr, crds_mov_ctgr, mo_estim_ngp, in_vivo=False):
    ksp_mov_ctgr_moco = np.copy(ksp_mov_ctgr)
    crds_ctgr_rot = np.zeros_like(crds_mov_ctgr)
    # shift mo estimates by one to the right
    mo_rp = np.zeros((160,6))
    mo_rp[1:] = mo_estim_ngp.transpose([1,0,2]).reshape((-1,6))[:-1]
    mo_estim_ngp_new = mo_rp.reshape((16,10,6)).transpose([1,0,2])
    
    Nt_navi = 10
    _, nt, N_groups, _ = crds_mov_ctgr.shape
    nr = 47
    for gr_ii in range(N_groups):
        for navi_ii in range(Nt_navi):
            dx, dy, dz, phi_x, phi_y, phi_z = mo_estim_ngp_new[navi_ii, gr_ii,:]

            # adjustments from the rotation implementation
            phi_x = -phi_x
            phi_y = -phi_y
            dx = -dx

            Rx = np.array([[1, 0, 0], [0, np.cos(phi_x), -np.sin(phi_x)], \
                  [0, np.sin(phi_x), np.cos(phi_x)]])
            Ry = np.array([[np.cos(phi_y), 0, -np.sin(phi_y)], [0, 1, 0], \
                  [np.sin(phi_y), 0, np.cos(phi_y)]])
            Rz = np.array([[np.cos(phi_z), -np.sin(phi_z), 0], \
                           [np.sin(phi_z), np.cos(phi_z), 0], [0, 0, 1]])

            # crds are in [-0.5, 0.5]
            begin = nr*(navi_ii)
            tr_corr = np.arange(begin,begin+nr,1)
            crds_c_tr_rot = Rx @ Ry @ Rz @ crds_mov_ctgr[:,:,gr_ii,tr_corr].reshape((3, -1))
            crds_ctr_rot = crds_c_tr_rot.reshape((3, nt, nr))
            crds_ctgr_rot[:,:,gr_ii,tr_corr] = crds_ctr_rot

            ph_ramp_x_ctr = np.tile(np.exp( -2j * np.pi * dx * crds_ctr_rot[[0]] / N ), (nc, 1, 1))
            ph_ramp_y_ctr = np.tile(np.exp( -2j * np.pi * dy * crds_ctr_rot[[1]] / N ), (nc, 1, 1))
            ph_ramp_z_ctr = np.tile(np.exp( -2j * np.pi * dz * crds_ctr_rot[[2]] / N ), (nc, 1, 1))

            ksp_mov_ctgr_moco[:,:,gr_ii,tr_corr] *= ph_ramp_x_ctr * ph_ramp_y_ctr * ph_ramp_z_ctr
    return ksp_mov_ctgr_moco, crds_ctgr_rot