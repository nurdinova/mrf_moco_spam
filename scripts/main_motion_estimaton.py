#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# #### Matching 3 spirals acquired every 50 TRs to the dictionary generated from dummy2 scout
#


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
import time

import torch
import cupy as xp
import sigpy.mri as mri
import pickle, json
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'notebook')

#### Figure style
font = 12 
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.size'] = font 
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['ytick.labelsize'] = font
plt.rcParams['xtick.labelsize'] = font
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['grid.linewidth'] = 2
plt.rcParams['grid.linewidth'] = 2
plt.rcParams['image.cmap'] = 'gray'

sys.path.append("/local_mount/space/ladyyy/data/users/aizadan/Subspace_Recon/mrf_moco_spam/src/")
from dict_gen_gpu import dict_gen
from plot_utils import plot_3dimg, find_closest_in_dict, plot_motion_traj
from utils import load_data
from motion_smooth_fit import fit_smooth
from motion_estimation import gEVD_dictionary, compress_memmap_dictionary, match_navi

device = sp.Device(7) # GPU, CPU = -1
xp = device.xp
mvc = lambda x : sp.to_device(x, sp.cpu_device)
mvd = lambda x : sp.to_device(x, device)

def clean_gpu():
    with device:
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    
def norm_signals(ksp_ngct):
    # don't normalize navigators per signal
    norm_map = np.linalg.norm(np.abs(ksp_ngct), axis=(2, 3), keepdims=True)
    ksp_ngct_norm = ksp_ngct / norm_map        
    return ksp_ngct_norm, norm_map

def norm_min_max_zero(ksp_bmct, zero_ind):
    ksp_zero_mo_bct = ksp_bmct[:, zero_ind]
    scale_factor_b = np.max(np.abs(ksp_zero_mo_bct), axis=(-1, -2), keepdims=True) # - np.min(np.abs(ksp_zero_mo_bct), axis=(-1, -2))
    return ksp_bmct / scale_factor_b[:, None]



# %%
date_tag = '240826'
dataset_number = 0

JSON_FILENAME = os.path.join('/local_mount/space/ladyyy/data/users/aizadan/Subspace_Recon/dictgator/', 'data_info.json')
with open(JSON_FILENAME, 'r') as json_data:
    metadata = json.load(json_data)[date_tag][dataset_number]

dir_matching = metadata['dir_matching']
dir_figure = metadata['dir_figures']
saveDIR = metadata['saveDIR']

data_path_new = metadata['recon_data_path']
dir_diff_poses = [f'Series{series_number}/' for series_number in metadata['Series']]

# %%
# parameters
generate_dictionary = False
generate_dB0_dictionary = False

do_SVD_dictionary = False # find the compression basis
SVD_comps_per_direction = False
do_coil_compr = True
select_coils = False

SVD_time = True
coils_to_select = [1, 2, 4, 9, 27]
Nt_up = 3 * 1658 # crop full signals in time
Nt_new_lo = 0
Nt_new_up = 120

match_mode = 'DictMatch' # 'DictMatch' or 'MLP'
match_difference = False 
calibrate_wrt_static = True # either subtract difference to static OR gEVD on static if do_SVD_dictionary
save_match = False

scout = "dummy"
matching = "interTR" #  "intergroup"
N_navi_tr = 6 # number+ of spiral navigators
select_spirals = True
navis_to_use = [1, 2, 3, 4, 5, 6, 8, 9]
svd_compr_method = 'pinv_SVD'
save_suffix = f'fd_coilSVD_tgEVD_6spir_dictGT_{svd_compr_method}_static_all_{match_mode}_test'



# %%
# load all the dataa
for ser_ii in range(len(dir_diff_poses)):
    data = load_data(dir_diff_poses[ser_ii], data_path=data_path_new, scout=scout, \
                              matching=matching, coil_compr=False, n_spirals=N_navi_tr)
    if ser_ii == 0:
        ksp_navi_ngc_tr = np.copy(data[3])
    else:
        ksp_navi_ngc_tr = np.concatenate((ksp_navi_ngc_tr, data[3]), axis=1)
print(ksp_navi_ngc_tr.shape)

img_dummy2_xyz, sens_dummy_xyzc, coords_navi_c_tr, _, img_navi_end_xyzp, img_dummy2_xyzc = \
                load_data(dir_diff_poses[0], data_path=data_path_new, scout=scout, \
                          matching=matching, coil_compr=False, n_spirals=N_navi_tr)
if select_coils:
    
    N_ch = len(coils_to_select)
    sens_dummy_xyzc = sens_dummy_xyzc[..., coils_to_select] #* np.exp(1j * phase_diff_xyz[..., None])  
    ksp_navi_ngc_tr = ksp_navi_ngc_tr[:, :, coils_to_select]
    
Nt_navi, N_GR, N_ch, N_t = ksp_navi_ngc_tr.shape
    
if select_spirals:
    spirals_to_select = [2, 3, 4]
    
    coords_navi_ctr = coords_navi_c_tr.reshape((3, -1, N_navi_tr))
    coords_navi_c_tr = coords_navi_ctr[..., :1658, spirals_to_select].reshape((3, -1))
    
    ksp_navi_ngctr = ksp_navi_ngc_tr.reshape((Nt_navi, N_GR, N_ch, -1, N_navi_tr))
    ksp_navi_ngc_tr = ksp_navi_ngctr[..., :1658, spirals_to_select].reshape((Nt_navi, N_GR, N_ch, -1))
#     ksp_navi_ngc_tr = (-1j) * ksp_navi_ngc_tr
    
    N_navi_tr = len(spirals_to_select)
    N_t = ksp_navi_ngc_tr.shape[-1]

sens_dummy_cxyz = sens_dummy_xyzc.transpose((3, 0, 1, 2)) 
print('sens_dummy_cxyz shape ', sens_dummy_cxyz.shape)

print(f'ksp_navi_ngct {ksp_navi_ngc_tr.shape}') 
print(f'coords_navi_c_tr {coords_navi_c_tr.shape}')
N_t_spiral = N_t // 3


# %%
if do_coil_compr:
    with device:
        N_ch_svd = 5
        u_rtt = cfl.readcfl(f'{data_path_new}{dir_diff_poses[0]}CC_Vc_full')
        print('u_rtt shape', u_rtt.shape)

        u_t_H = mvd(u_rtt[:, :N_ch_svd])
        sens_dummy_xyzc = xp.matmul(mvd(sens_dummy_xyzc), u_t_H[None, None]).get()
        print(f'sens_dummy_xyzc shape {sens_dummy_xyzc.shape}')
        sens_dummy_cxyz = sens_dummy_xyzc.transpose((-1, 0, 1, 2))

        ksp_navi_ngc_tr = xp.matmul(mvd(ksp_navi_ngc_tr).transpose((0, 1, 3, 2)), u_t_H[None, None]).get().transpose((0, 1, 3, 2))
        N_ch = N_ch_svd
        print(f'ksp_navi_ngc_tr shape {ksp_navi_ngc_tr.shape}')

# %%
mo_GT = np.load(f'{data_path_new}dictgator/motion_GT_all.npy')[:N_GR]
print(f'motion GT data {mo_GT.shape}')

# %%
# navigators from the static dataset for calibration
calib_by_mean = False

# normalize - calibrate - scale back for ng
if calibrate_wrt_static:
    group_to_calib = np.arange(16)
    
    ksp_spinav_ctgn = cfl.readcfl(data_path_new+dir_diff_poses[0]+'ksp_spinav_ctgr_noCC')
    if select_coils:
        ksp_spinav_ctgn = ksp_spinav_ctgn[coils_to_select]
    elif do_coil_compr:
        with device:
            ksp_spinav_ctgn = xp.matmul(mvd(ksp_spinav_ctgn).transpose((1, 2, 3, 0)), u_t_H[None, None]).get().transpose((3, 0, 1, 2))

    if matching == 'intergroup':
        ksp_spinav_ctgn = ksp_spinav_ctgn[..., -3:]
    ksp_spinav_ctgn = ksp_spinav_ctgn[:, :1658, ...]
    print('ksp_ctgn ', ksp_spinav_ctgn.shape)
        
    ksp_stat_ngctr = ksp_spinav_ctgn.reshape((N_ch, N_t_spiral, 16, Nt_navi, -1)).transpose((-2, 2, 0, 1, -1)) 
    if select_spirals:
        ksp_stat_ngctr = ksp_stat_ngctr[..., spirals_to_select]
    
    ksp_stat_ngct = ksp_stat_ngctr.reshape((Nt_navi, 16, N_ch, -1))
    ksp_stat_ngct = ksp_stat_ngct[:, group_to_calib, ...]
    ksp_stat_ngct /= ksp_stat_ngct.size
    print('ksp_stat_ngct shape ', ksp_stat_ngct.shape)
    

# %%
Phi_navi_ct = np.squeeze(cfl.readcfl(data_path_new + dir_diff_poses[0]+'Phi_spinav_cc')).T
if select_spirals:
    Phi_navi_ctr = Phi_navi_ct.reshape((5, Nt_navi, -1))
    Phi_navi_ct = Phi_navi_ctr[..., spirals_to_select].reshape((5, -1))
    
img_dummy2_xyzt = np.matmul(img_dummy2_xyzc, Phi_navi_ct)
print(f'img_dummy2_xyzt shape {img_dummy2_xyzt.shape}')

# %%
dict_name = 'fd_coilgEVD_6spir_pinv_SVD_static_all_DictMatch'
with open(f"{saveDIR}input_gendict_{dict_name}.pkl", "rb") as file:
    input_gendict = pickle.load(file)

# %%
clean_gpu()
n_params = 6 # motion DOF

if matching == 'intergroup':
    N_coef = 1
    dictionary = 'scout'    
elif matching == 'interTR':
    N_coef = 5 
    dictionary = 'qscout' 

if do_SVD_dictionary:
    n_pars_per_direction = 4
    motion_mp = np.stack(np.meshgrid(input_gendict["shifts_x"][:n_pars_per_direction], input_gendict["shifts_y"][:n_pars_per_direction], \
                                     input_gendict["shifts_z"][:n_pars_per_direction], input_gendict["angles_x"][:n_pars_per_direction], \
                                     input_gendict["angles_y"][:n_pars_per_direction], input_gendict["angles_z"][:n_pars_per_direction]), -1).reshape((-1, 6))


    N_mo_svd = motion_mp.shape[0]
    ind_to_sim = np.random.choice(motion_mp.shape[0], N_mo_svd - 1, replace=False)
    motion_to_sim_mp = np.zeros((N_mo_svd, 6))
    motion_to_sim_mp[1:] = motion_mp[ind_to_sim]
    
    dict_name = f'qdict_forSVD_{save_suffix}'
    input_gendict["dict_fullname"] = saveDIR + f'{dict_name}'

    tic = time.perf_counter()
    if generate_dictionary:
        with device:
            clean_gpu()
            input_gendict["input_img_xyzq"] = mvd(img_dummy2_xyzt[..., [-3]])
            input_gendict["device"] = device
            input_gendict["motion_mp"] =  mvd(motion_to_sim_mp[:20])

            input_gendict["sens_xyzc"] = mvd(sens_dummy_cxyz).transpose((1, 2, 3, 0))
            input_gendict["k_coords_ka"] = mvd(coords_navi_c_tr).astype(float).T
#             input_gendict["Phi_qnr"] = mvd(Phi_navi_ct).reshape((N_coef, -1, N_navi_tr))
            input_gendict["rot_center"] = "center"
            dict_gen(input_gendict, debug=0, coils_before_motion=False)
            
    toc = time.perf_counter()
    time_elapsed = toc - tic
    print(f"Finished! Total time = {time_elapsed:0.4f} seconds")
    print(f"Loading the {dictionary} Dictionary: {dict_name}")

    dict_ksp_mctq = np.memmap(input_gendict["dict_fullname"]+'.dat', dtype=np.complex64, 
                  mode='r', shape=(N_mo_svd, N_ch, N_t, Nt_navi))
    dict_mo_pars_mp = np.memmap(input_gendict["dict_fullname"]+'_moIdx.dat', dtype=np.float32, 
              mode='r', shape=(N_mo_svd, n_params))



# %%
dict_dB0_name = saveDIR + 'dict_dB0_1stSH_selcoils_set2_Ser7_pnct_3'

if do_SVD_dictionary and svd_compr_method in ['gEVD', 'pinv_SVD']:
    if not calibrate_wrt_static:
        dict_dB0_motion_npct = cfl.readcfl(dict_dB0_name)
        print(f'Using dB0-simulations for gEVD')

# %%
# # make sure dict_dB0 and dict_motion signals are on the same scale 
if do_SVD_dictionary:
    dict_ksp_scaled_nmct = norm_min_max_zero(dict_ksp_mctq.transpose((3, 0, 1, 2)), 0)
    if calibrate_wrt_static:
        ksp_stat_scaled_ngct = norm_min_max_zero(ksp_stat_ngct, 0)
    else:
        ksp_stat_scaled_ngct = norm_min_max_zero(dict_dB0_motion_npct.transpose((1, 0, 2, 3)), 0)
    print(f'in vivo signals shape {ksp_stat_scaled_ngct.shape}')
    print(f'simulate signals shape {dict_ksp_scaled_nmct.shape}')

# %%
if do_SVD_dictionary:
    spir = 1
    ch = 1
    func_to_plot = lambda x: np.real(x)
    func_to_plot_2 = lambda x: np.imag(x)
    for nii in [3, 5, 6]:
        for i in range(1):
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].plot(func_to_plot(ksp_stat_scaled_ngct[nii][i, ch, spir::N_navi_tr]), label='In vivo')
            axs[0].plot(func_to_plot(dict_ksp_scaled_nmct[nii][0, ch, spir::N_navi_tr]), label='Simulated')
            axs[0].set_title('Real')

            axs[1].plot(func_to_plot_2(ksp_stat_scaled_ngct[nii][i, ch, spir::N_navi_tr]), label='In vivo')
            axs[1].plot(func_to_plot_2(dict_ksp_scaled_nmct[nii][0, ch, spir::N_navi_tr]), label='Simulated')
            axs[1].set_title('Imag')
            axs[1].set_yticks([])

            for i in [0, 1]:
                axs[i].set_xlabel('Time')
                axs[i].legend() 
                axs[i].set_xlim([50, 500])
                axs[i].set_ylim([-0.1, 0.1])
            plt.suptitle(f'nii {nii}')
            plt.tight_layout()

# %%
## exclude outlier

groups_to_use = []
for nii in range(10):
    if calibrate_wrt_static:
        groups_to_use.append(np.arange(16))
        if (nii == 1):
            groups_to_use[-1] = np.delete(groups_to_use[-1], [1, 2])
        if (nii in [8, 9]):
            groups_to_use[-1] = np.delete(groups_to_use[-1], 7)
    else:
        groups_to_use.append(np.arange(N_mo_svd))


# %%
# time-compression basis
Nt_up = 4974 // 3
do_crop_signals = False
          
if do_SVD_dictionary:
    dict_ksp_scaled_nmct = norm_min_max_zero(dict_ksp_mctq.transpose((3, 0, 1, 2)), 0)

    if do_crop_signals:
        dict_ksp_scaled_nmct = dict_ksp_scaled_nmct[..., :Nt_up*N_navi_tr]
        ksp_stat_scaled_ngct = ksp_stat_scaled_ngct[..., :Nt_up*N_navi_tr]

    u_rtt = np.zeros((Nt_navi, N_ch, N_t, N_t), dtype=np.complex64)
    s_rt = np.zeros((Nt_navi, N_ch, N_t), dtype=np.complex64)

    for nii in navis_to_use:
        clean_gpu()
        with device:
#             for ch_ii in range(N_ch):
            if svd_compr_method in ['gEVD', 'pinv_SVD']:
                dict_dB0_npct = mvd(ksp_stat_scaled_ngct[[nii]]).transpose((1, 2, 0, 3))
            else:
                dict_dB0_npct = None

            dict_ksp_gpu = mvd(dict_ksp_scaled_nmct[[nii]]).transpose((1, 2, 0, 3))

            dict_dB0_npct = dict_dB0_npct[groups_to_use[nii], ...]
            dict_dB0_npct = xp.concatenate((dict_ksp_gpu[[0], ...], dict_dB0_npct), axis=0)

            print('A and B sizes', dict_ksp_gpu.shape, dict_dB0_npct.shape)
            result_svd, _ = gEVD_dictionary(dict_ksp_gpu, dict_dB0_npct, xp=xp, \
                                            method=svd_compr_method, demean_data=True, \
                                            unit_variance=False, scale_variance=True)

            if svd_compr_method in ['SVD', 'pinv_SVD']:
                s_rt[nii] = np.tile(result_svd[1].get(), (N_ch, 1))
                u_rtt[nii] = np.tile(result_svd[0].get(), (N_ch, 1, 1))
            else:
                s_rt[nii] = result_svd[0]
                u_rtt[nii] = result_svd[1]

    cfl.writecfl(f'{data_path_new}dictgator/u_nctt_{save_suffix}', u_rtt)
    cfl.writecfl(f'{data_path_new}dictgator/s_nct_{save_suffix}', s_rt)

# %%
# u_nrtt = cfl.readcfl(f'./saved_files_240802_spinav/u_rtt_6spir_gEVD_dB0_1stSH_DictMatch')
suffix_basis = 'fd_coilSVD_tgEVD_6spir_pinv_SVD_static_all_DictMatch'
u_nrtt = cfl.readcfl(f'{data_path_new}dictgator/u_nctt_{suffix_basis}')
Nt_up = 4974 // 3
N_ct = Nt_up * 1
u_nrtt.shape

# %%
# include GT in the dict
input_gendict = {}
input_gendict["angles_x"] = np.arange(-0.5, 1., 0.5) * np.pi / 180
input_gendict["angles_y"] = np.arange(-1, 1.5, 0.5) * np.pi / 180
input_gendict["angles_z"] = np.arange(-2.5, 0.5, 0.5) * np.pi / 180

input_gendict["shifts_x"] = np.arange(-0.4, 0.4, 0.2)
input_gendict["shifts_y"] = np.arange(-0.2, 0.4, 0.2)
input_gendict["shifts_z"] = np.arange(-0.6, 0.4, 0.2)

n_x = input_gendict["shifts_x"].shape[0]
n_y = input_gendict["shifts_y"].shape[0]; n_z = input_gendict["shifts_z"].shape[0]
n_Rx = input_gendict["angles_x"].shape[0]; n_Ry = input_gendict["angles_y"].shape[0]
n_Rz = input_gendict["angles_z"].shape[0]

motion_mp = np.stack(np.meshgrid(input_gendict["shifts_x"], input_gendict["shifts_y"], \
                                     input_gendict["shifts_z"], input_gendict["angles_x"], \
                                     input_gendict["angles_y"], input_gendict["angles_z"]), -1).reshape((-1, 6))
motion_mp = np.concatenate((motion_mp, mo_GT), axis=0)

input_gendict["motion_mp"] = motion_mp

n_pars = motion_mp.shape[0] 
print("Number Motion states in dict ", n_pars)
input_gendict['n_pars_total'] = n_pars
 
with open(f"{saveDIR}input_gendict_{save_suffix}.pkl", "wb") as file:
    pickle.dump(input_gendict, file)

# %%
if do_SVD_dictionary:
#     del input_gendict["motion_mp"]
    del input_gendict["Phi_qnr"]
Nt_new_up = 120

for nii in navis_to_use:
    with device:
        u_rtt = mvd(u_nrtt[nii])
        if do_coil_compr:
            u_t = u_rtt
        else:
            u_t = u_rtt.transpose((1, 0, 2)).reshape((3*N_ct, N_ct))
        u_t_H = u_t[..., Nt_new_lo:Nt_new_up]

        dict_name = f'qdict_full_nii{nii}_{save_suffix}'
        # dict_name = 'qdict_selcoils_set2_Ser7_n8_ctgEVD_static_120_2test'
        input_gendict["dict_fullname"] = saveDIR + f'{dict_name}'

        tic = time.perf_counter()

        if generate_dictionary:
            clean_gpu()        
            input_gendict["input_img_xyzq"] = mvd(img_dummy2_xyzt[..., [3*nii]])
            input_gendict["device"] = device
            input_gendict["sens_xyzc"] = sens_dummy_cxyz.transpose((1, 2, 3, 0))
            input_gendict["k_coords_ka"] = coords_navi_c_tr[:, :N_navi_tr*Nt_up].astype(float).T
    #         input_gendict["Phi_qt"] = mvd(Phi_navi_ct[:, nii * 3 : (nii + 1) * 3])
            input_gendict["v_ct_tnew"] = u_t_H
            input_gendict["rot_center"] = "center"
            dict_gen(input_gendict, debug=0, coils_before_motion=False)

            toc = time.perf_counter()
            time_elapsed = toc - tic
            print(f"Finished! Total time = {time_elapsed:0.4f} seconds")

# %%
Nt_compr = 120
N_ch_compr = N_ch

def maes(mo_1, mo_2):
    mae_translations = np.mean(np.abs(4 * (mo_1[:, :3] - mo_2[:, :3])), axis=-1)
    mae_rotations = np.mean(np.abs(180 / np.pi * (mo_1[:, 3:] - mo_2[:, 3:])), axis=-1)
    return mae_translations, mae_rotations 
    
def run_matching(Nt_new_lo, Nt_new_up, lamda_kalman=0, n_minima=1):
    motion_Est = np.zeros((Nt_navi, N_GR, 6))
    mae_translation_bn = np.zeros((Nt_navi, N_GR))
    mae_rotation_bn = np.zeros((Nt_navi, N_GR))
    for nii in navis_to_use:
        suffix_dict = 'fd_coilSVD_tgEVD_6spir_dictGT_pinv_SVD_static_all_DictMatch'
        dict_name = f'qdict_full_nii{nii}_{suffix_dict}'
        input_gendict["dict_fullname"] = saveDIR + f'{dict_name}'
        n_pars = input_gendict['n_pars_total']

        dict_ksp_mctq = np.memmap(input_gendict["dict_fullname"]+'.dat', dtype=np.complex64, 
                      mode='r', shape=(n_pars, N_ch_compr, Nt_compr, 1))
        dict_mo_pars_mp = np.memmap(input_gendict["dict_fullname"]+'_moIdx.dat', dtype=np.float32, 
                  mode='r', shape=(n_pars, n_params))

        save_tag = f's{7+dataset_number}_nii{nii}_{save_suffix}_{matching}_comp{Nt_new_lo}_{Nt_new_up}'

        with device:
            u_t = u_nrtt[nii]
            if not do_coil_compr:
                u_t = u_t.transpose((1, 0, 2)).reshape((N_t*N_ch, N_ch*N_t//3))
                u_t = u_t[None]
            u_t_H_to_match = mvd(u_t[..., Nt_new_lo:Nt_new_up])
            ksp_navi_ngc_tr_to_match = mvd(ksp_navi_ngc_tr[[nii], ..., :N_t])
            ksp_navi_ngc_tr_to_match = ksp_navi_ngc_tr_to_match[:, :]
            if do_coil_compr:
                ksp_navi_compr_ngct = xp.zeros((1, N_GR, N_ch_compr, Nt_new_up-Nt_new_lo), dtype=np.complex64)
                for ch_ii in range(N_ch_compr):
                    ksp_navi_compr_ngct[:, :, ch_ii] = xp.matmul(u_t_H_to_match[ch_ii].transpose((1, 0))[None, None], ksp_navi_ngc_tr_to_match[:, :, ch_ii, :, None])[..., 0] 
                ksp_navi_ngc_tr_to_match = ksp_navi_compr_ngct
                u_t_H_to_match = None
                
        if match_mode == 'DictMatch':
            dict_ksp_mctq = dict_ksp_mctq[:, :, Nt_new_lo:Nt_new_up, :]
            model_name_all_params = None
        elif match_mode == 'MLP':
            labels = ['dx', 'dy', 'dz', 'phy_x', 'phi_y', 'phi_z']
            model_name_all_params = []
            for mo_par in range(6):
                writer_name = f'dict_{labels[mo_par]}_woNoise_gEVD_BN_Linf_STDnorm_Adam'
                model_name_all_params.append(f'mlp_regr_full_split0.8_{writer_name}')
        
        if nii not in [0, 1, 8] and lamda_kalman:
            mo_prev_gp = motion_Est[nii-1, ...]
        else:
            mo_prev_gp = None
        with device:
#             print(f'ksp_ngct and dict_mct shape {ksp_navi_ngc_tr_to_match.shape, dict_ksp_mctq.shape}')
            res_matching_dict = match_navi(ksp_navi_ngc_tr_to_match, dict_ksp_mctq[..., [0]],\
                                           dict_mo_pars_mp, u_t_H_to_match, input_gendict, \
                                           mo_GT=mo_GT[:], \
                                           img_navi_xyzp=img_navi_end_xyzp, nn_model_name=model_name_all_params, match_mode=match_mode, \
                                           match_difference=False, save_match=False, \
                                           dir_figure=dir_figure, dir_matching=dir_matching, save_tag=save_tag, \
                                           smoothfit_motion_noGT=False, \
                                           smoothfit_motion_with_GT=False, \
                                           smoothfit_motion_GT=False, verbose=0, \
                                           smooth_match=True, lamda_kalman=lamda_kalman, mo_prev_gp=mo_prev_gp, n_minima=n_minima)
        motion_Est[nii, ...] = np.copy(res_matching_dict['motion_Est'])
        mae_translation_bn[nii], mae_rotation_bn[nii] = maes(motion_Est[nii, :N_GR], \
                                                               mo_GT[:N_GR])
    return motion_Est, mae_translation_bn, mae_rotation_bn


# %%

def plot_violins(mae_translation_bn, mae_rotation_bn, save=False):
    mean_energy_ksp_navi_n = np.mean(np.linalg.norm(ksp_navi_ngc_tr, axis=(-1, -2)), axis=1)
    mean_energy_ksp_navi_n /= np.max(mean_energy_ksp_navi_n)

    error_vs_energy = {"abs_error": [], "type": [], "navi_ii": []}
    for nii in range(Nt_navi):
        # Add translation errors
        error_vs_energy["type"].extend(["Translation"] * N_GR)
        error_vs_energy["abs_error"].extend(mae_translation_bn[nii, :N_GR])
        error_vs_energy["navi_ii"].extend([nii] * N_GR)

        # Add rotation errors
        error_vs_energy["type"].extend(["Rotation"] * N_GR)
        error_vs_energy["abs_error"].extend(mae_rotation_bn[nii, :N_GR])
        error_vs_energy["navi_ii"].extend([nii] * N_GR)


    plt.figure()
    sns.violinplot(x='navi_ii', y='abs_error', hue='type', data=error_vs_energy, split=True)
    plt.plot(np.arange(Nt_navi), mean_energy_ksp_navi_n, label='Energy')
    plt.legend()
    plt.ylabel('')
    plt.title('MAE, mm or deg')
    plt.xlabel('Contrast index')
    plt.tight_layout()
    if save:
        plt.savefig(f'{dir_figure}violins_{save_suffix}')

# %%
Nt_new_lo = 0
Nt_new_up = 70
Nt_navi_to_use = Nt_navi
N_GR = 48
n_minima = 5
motion_GT_gnp = np.tile(mo_GT[:, None, :], (1, Nt_navi_to_use, 1))

# for lamda in [0, 1e4]:
n_minima_test = np.arange(1, 40, 2)
n = len(n_minima_test)
mae_transl = np.zeros((n, ))
mae_rot = np.zeros((n, ))
for i in range(n):
    motion_Est, mae_translation_ng, mae_rotation_ng = run_matching(Nt_new_lo, Nt_new_up, 1e4, n_minima_test[i])
    motion_Est[7] = 0.5 * (motion_Est[8] + motion_Est[6])
    motion_Est[0] = np.copy(motion_Est[1])
    mae_transl[i] = np.mean(mae_translation_ng)
    mae_rot[i] = np.mean(mae_rotation_ng)
#     plot_violins(mae_translation_ng, mae_rotation_ng)

plt.figure()
plt.plot(n_minima_test, mae_rot, '-o', label='MAE rotations, deg')
plt.plot(n_minima_test, mae_transl, '-o', label='MAE translations, mm')
plt.legend()
plt.xlabel('n_minima')
plt.title(f'lamda_kalman = 1e4')
plt.tight_layout()

# %%
motion_Est, mae_translation_ng, mae_rotation_ng = run_matching(Nt_new_lo, Nt_new_up, 1e2, 25)
motion_Est[7] = 0.5 * (motion_Est[8] + motion_Est[6])
motion_Est[0] = np.copy(motion_Est[1])
plot_violins(mae_translation_ng, mae_rotation_ng)

motion_Est_gnp = motion_Est.transpose((1, 0, 2))
res_match = {'motion_Est': motion_Est_gnp.reshape((-1, 6)), \
             'motion_GT': motion_GT_gnp.reshape((-1, 6))}

figname_traj = f"{dir_figure}resMatch_motion_{save_tag}"
plot_motion_traj(res_match, img_resolution=4, N_GR=N_GR, Nt_navi=Nt_navi_to_use, \
                 save=False, figname='', plot_smooth_noGT=False, \
                 figsize=(10, 5), translation_lims=[-3, 3], rotation_lims=[-3, 3])
#     nii = 9
#     res_match = {'motion_Est': motion_Est[nii, ...].reshape((N_GR, 6)), \
#                  'motion_GT': mo_GT.reshape((N_GR, 6))}
#     figname_traj = dir_figure+f"resMatch_motion_{save_tag}_nii{nii}"
#     plot_motion_traj(res_match, img_resolution=4, N_GR=N_GR, Nt_navi=1, save=True, \
#                      figname=figname_traj, translation_lims=[-3, 3], rotation_lims=[-3, 3])

# %%
for nii in navis_to_use:
    res_match = {'motion_Est': motion_Est[nii, ...].reshape((N_GR, 6)), \
                 'motion_GT': mo_GT.reshape((N_GR, 6))}
    print(nii)
    figname_traj = dir_figure+f"resMatch_motion_{save_tag}_nii{nii}"
    plot_motion_traj(res_match, img_resolution=4, N_GR=N_GR, Nt_navi=1, save=True, \
                     figname=figname_traj, translation_lims=[-3, 3], rotation_lims=[-3, 3])
