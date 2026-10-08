#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ### Matching of all groups with 3 spirals
#
# ### Pay attention that matching and data is on the same device, otherwise you'll notice the loss curve is weird and matching doesn't work.


# %% [markdown]
# Found a bug in reshaping the data: 3 spirals were mixed, so if we have a Nt = 300, hundred timepoints forom each spiral were interleaved. Doesn't affect the results tho as the mistake is consistent!

# %%
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys 
sys.path.append("/usr/local/app/bart/bart-0.6.00/python/")
import sigpy as sp
import cfl 
import sigpy.plot as pl

import cupy as xp
import sigpy.mri as mri
import pickle, json
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'notebook')

sys.path.append("/local_mount/space/mayday/data/users/aizadan/dictgator/")
from src.simulate_Nav_signal import load_data
from src.analysis.plot_dictMatch import plot_motion_traj, plot_3dimg
from src.helper_moco_subspace_recon import motion_corr_interTR, plot_imgs_N_diff

sys.path.append("/local_mount/space/mayday/data/users/aizadan/mrf_moco_spam/src/")
from motion_correction import kspace_rigid_moco

device_num = 6
device = sp.Device(device_num) # GPU, CPU = -1
xp = device.xp
mvc = lambda x : sp.to_device(x, sp.cpu_device)
mvd = lambda x : sp.to_device(x, device)

def clean_gpu():
    with device:
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


# %%
date_tag = '250307'
JSON_FILENAME = os.path.join('/local_mount/space/mayday/data/users/aizadan/dictgator/', 'data_info.json')
with open(JSON_FILENAME, 'r') as json_data:
    metadata = json.load(json_data)[date_tag][0]

dir_matching = metadata['dir_matching']
dir_figure = metadata['dir_figures'] + '/moco/'
os.makedirs(dir_figure, exist_ok=True)
saveDIR = metadata['saveDIR']

data_path_new = metadata['recon_data_path']
dir_diff_poses = [f'Series{series_number}/' for series_number in metadata['Series']]

save_suffix = 'fd_coilSVD_ctgEVD_6spir_dictGT_nii12_pinv_SVD_static_all_DictMatch_test7_u_allnavi'
voxel_size = 4
N_GR_dataset = 24
N_GR = N_GR_dataset * len(dir_diff_poses)
Nt_navi = 13

# %%
# test
with open(dir_matching+f'resMatch_{save_suffix}.pkl', 'rb') as f:
    res_matching = pickle.load(f)  

# %%

# %%
N_image = 220
suffix_data = 'cc'

for series_ii in [1]:
    recon_dir_mov = f'{data_path_new}/{dir_diff_poses[series_ii]}/'
    dir_moco_data = f'{data_path_new}/{dir_diff_poses[series_ii]}/tmp_moco/'
    os.makedirs(dir_moco_data, exist_ok=True)

    ksp_mov_ctgr = np.transpose(cfl.readcfl(recon_dir_mov+f'ksp_full_{suffix_data}')[0,...,0,:], \
                                                      [2, 0, 1, 3])
    crds_mov_ctgr = np.squeeze(cfl.readcfl(recon_dir_mov+f'crds_full_{suffix_data}')).astype(np.float32)
    sens_mov_xyzc = cfl.readcfl(recon_dir_mov+f'sens_full_xyzc_{suffix_data}')
    img_mov_xyzc = cfl.readcfl(recon_dir_mov+f'recon_full_xyzc_cc')
    print(ksp_mov_ctgr.shape, crds_mov_ctgr.shape, sens_mov_xyzc.shape)    
    Phi = np.squeeze(cfl.readcfl(recon_dir_mov+f'Phi_full_{suffix_data}'))
    
    TR_to_plot = 125
    Phi_c_200 = Phi[TR_to_plot]
    img_mov_xyz = img_mov_xyzc @ Phi_c_200

    
    with open(dir_matching+f'resMatch_{save_suffix}.pkl', 'rb') as f:
        res_matching = pickle.load(f)  

    motion_ngp = - res_matching['motion_GT'].reshape((N_GR, Nt_navi, 6)).transpose((1, 0, 2))
    motion_ngp = motion_ngp[:, N_GR_dataset*series_ii:N_GR_dataset*(series_ii+1), :]
    motion_ngp[..., :3] *= voxel_size
    print('motion_ngp shape', motion_ngp.shape)
    
    ksp_moco_ctgr_new, crds_moco_ctgr_new = kspace_rigid_moco(ksp_mov_ctgr, crds_mov_ctgr, \
                                                        motion_ngp, N_image, 436//13, device=device)

#     recon_moco_dict_xyzc = run_subspace(ksp_moco_ctgr, crds_moco_ctgr, sens_mov_xyzc, Phi, \
#                                         device_num=device_num, maxiter=30, \
#                                         recon_dir=moco_dir, recon_name=filename, lowres_recon=False)
#     recon_moco_dict_xyzc = cfl.readcfl(moco_dir+'recon_'+filename)
#     img_moco_dict_xyz = recon_moco_dict_xyzc @ Phi_c_200

#     plot_imgs_N_diff(img_moco_dict_xyz, img_moco_dict_xyz, \
#                      N_image, vmax_factor=[1., 0.8, 1.0], \
#                      diff_scale=10, labels=['Moving', 'MoCo AFNI, intergroup'], save=True, \
#                      save_name=dir_figure+filename)

# %%
