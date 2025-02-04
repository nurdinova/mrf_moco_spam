import sigpy as sp
import sys 
sys.path.append("/usr/local/app/bart/bart-0.6.00/python/")
import sigpy as sp
import cfl 
import numpy as np

def clean_gpu(device):
    xp = device.xp
    with device:
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
def load_data(dir_pose, data_path='', \
              scout='dummy', matching='intergroup', coil_compr=False, n_spirals=3):
    ### Load q-scout data, the data is coil-compressed
    if data_path == '':
        data_path = '/local_mount/space/ladyyy/data/users/aizadan/scans_/moco_data_20231013/Recon_clean/';

    suffix_cc = '_cc' if coil_compr else '_noCC'
    if scout == "dummy":
        sens_dummy2_xyzc = cfl.readcfl(data_path+dir_pose+'sens_dummy2_xyzc_noCC')
        print('sens_dummy2_xyzc shape', sens_dummy2_xyzc.shape)
        N_ch = sens_dummy2_xyzc.shape[-1]

        # low-res image navigators at the end of each group
        img_navi_end_xyzp = cfl.readcfl(data_path+dir_pose+"recon_navi_xyzc_cc")
        print('img_navi_end_xyzp shape ', img_navi_end_xyzp.shape)
        
        img_dummy2_xyz = cfl.readcfl(data_path+dir_pose+"img_dummy2_xyzt_cc")[..., -n_spirals]
        print('img_dummy2_xyz shape ', img_dummy2_xyz.shape)
        img_dummy2_xyzc = cfl.readcfl(data_path+dir_pose+'recon_dummy2_xyzc_cc')
        print('img_dummy2_xyzc shape ', img_dummy2_xyzc.shape)
        
        # coordinate, time, TR and group dimensions
        coords_navi_c_tr = 55 * np.transpose(np.squeeze(cfl.readcfl(data_path+\
                                        dir_pose+'crds_spinav_tcr_noCC')),\
                                        [1, 0, 2]).reshape((3, -1)).astype(float)
        print('coords_navi_c_tr shape ', coords_navi_c_tr.shape)


        ksp_navi_ctgr = np.squeeze(cfl.readcfl(data_path+dir_pose+'ksp_spinav_ctgr_noCC'))
        if matching == 'intergroup':
            ksp_navi_ctgr = ksp_navi_ctgr[..., -n_spirals:]
        N_ch, N_t, N_GR = ksp_navi_ctgr.shape[:3]
        ksp_navi_ctgnr = ksp_navi_ctgr.reshape((N_ch, N_t, N_GR, -1, n_spirals)) # n - nav TR
        Nt_navi = ksp_navi_ctgnr.shape[-2]
        ksp_navi_ngctr = ksp_navi_ctgnr.transpose([3, 2, 0, 1, 4])
        ksp_navi_ngc_tr = ksp_navi_ngctr.reshape((Nt_navi, N_GR, N_ch, -1))
        print(f'ksp_navi_ngc_tr shape {ksp_navi_ngc_tr.shape}')
        # nuFFT normalizes the output by the number of samples, normalize the in vivo nav once in the beginning
        ksp_navi_ngc_tr /= ksp_navi_ngc_tr.size

    print(f'Scout data and sequence params are loaded \n')
    return img_dummy2_xyz, sens_dummy2_xyzc, coords_navi_c_tr, ksp_navi_ngc_tr, img_navi_end_xyzp, img_dummy2_xyzc
