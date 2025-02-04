import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys 
sys.path.append("/usr/local/app/bart/bart-0.6.00/python/")
import cfl 
from scipy.ndimage import rotate
from  bart import bart
import sigpy as sp
import sigpy.plot as pl
import nibabel as nib
import time
import cupy as cp
import pickle
from plot_utils import plot_3dimg, find_closest_in_dict, plot_motion_traj
from motion_smooth_fit import fit_smooth
from dict_match_gpu import match_toDict_gpu
from external import call_afni, call_flirt
from scipy.linalg import eig


def analyze_match_interscan(ksp_nav_gct, dict_mo_mct, dict_mo_idx_full, groups, img_vols_gxyz, motion_GT=None, \
                            loss="l1", subspace_u=None, return_matched=False, device=None, plot=True, verbose=0, \
                            Phi=None, T_ind=0, chunksize=None, ksp_dict_zero=None, ksp_nav_zero=False, \
                            mo_prev_gp=None, lamda_kalman=0., n_minima=1):
    """
    Estimate motion with dictionary matching amd compare the results to registration.
    Analysis will be done per group and for all coil channles/components.
    One dictionary for all groups/TRs/poses in dim = 0 of the ksp_nav and image volumes.
    
    ksp_nav_gct : array of dim (N_groups, N_chan, N_t)
    dict_mo_full : (N_mo_pars, N_chan, N_t)
    TRs : list
    
    img_vols_gxyz : array (N_groups, nx, ny, nz)
        images for registration
        volume 0 is the reference
        Note that img_vols_gxyz.shape[0] = ksp_nav_gct.shape[0] + 1 as it images contain the reference
 
    
    Return:
        dict : 
            "motion_GT": (N_TRs,N_comp,6) 
            "motion_Est": (N_TRs,N_comp,6)
            "L1_res" : (N_TRs,N_comp)
    
    """
    device = sp.get_device(ksp_nav_gct)
        
    if chunksize is None:
        chunksize = 256*4
    
    N_GR = len(groups)
    N_ch, N_t = ksp_nav_gct.shape[-2:]
        
    motion_Est = np.zeros((N_GR, 6))
    if return_matched:
        ksp_matched = np.zeros((N_GR, N_ch, N_t), dtype=np.complex)
    else:
        ksp_matched = None
    
    if (motion_GT is None):
        img_xyzp = np.transpose(img_vols_gxyz, [1, 2, 3, 0])
         # afni outputs motion from pose_ii to pose 0 - we need the opposite
        _, motion_GT = call_afni(img_xyzp, pre_saved=False, save_out=False) # outputs mm and deg
        motion_GT[:, 3:] *= np.pi/180
        motion_GT = motion_GT[:, :] # throw zeros row out
        motion_GT[:, 4] *= -1 # AFNI's output is CCW, my operator rotates CW
        
    L1_res = np.zeros((N_GR, 1))
    for group_ii, group in enumerate(groups):
        if verbose:
            print(f"Group = {group}")
        if mo_prev_gp is None:
            mo_prev = None
        else:
            mo_prev = mo_prev_gp[group]
        
        matched, loss_l1, motion_Est[group_ii,:] = match_toDict_gpu(ksp_nav_gct[group,:,:], dict_mo_mct, dict_mo_idx_full,  \
                                                motion_GT[group], loss=loss, chunksize=chunksize, \
                                                plot=plot, verbose=verbose, subspace_u=subspace_u, Phi=Phi, T_ind=T_ind, \
                                                ksp_dict_zero=ksp_dict_zero, ksp_nav_zero=ksp_nav_zero, mo_prev=mo_prev, \
                                                                    lamda_kalman=lamda_kalman, n_minima_chunk=n_minima)
        if return_matched:
            ksp_matched[group_ii] = np.copy(matched)
        L1_res[group_ii] = (np.sort(loss_l1)[0])
        
    return {"motion_GT":motion_GT, "motion_Est":motion_Est, "L1_res":L1_res, "matched_matrix": ksp_matched}

def match_navi(ksp_navi_ngc_tr, dict_ksp_mctq, dict_mo_pars_mp, \
               u_t_H, input_gendict, mo_GT=None, img_navi_xyzp=None, match_mode='DictMatch', \
               nn_model_name=None, match_difference=False, save_match=False, \
               dir_figure='./', dir_matching='', save_tag='', smoothfit_motion_noGT=False, \
               smoothfit_motion_with_GT=False, smoothfit_motion_GT=False, verbose=0, \
               smooth_match=False, lamda_kalman=0., mo_prev_gp=None, n_minima=1):
    """
    ksp_navi_ngc_tr : on device
    dict_ksp_mctq : memmap on cpu
        time dimension is expected to be compressed
    dict_mo_pars_mp : memmap on cpu
    u_t_H : on device
    img_navi_xyzp : on cpu
    
    Nt_up : int
        crop the full signals
    smooth_match : bool
        add L2-regularization along the navigator contrast dimension
        
    """
    device = sp.get_device(ksp_navi_ngc_tr)
    xp = device.xp
    
    Nt_navi_match, N_GR, N_ch, Nt = ksp_navi_ngc_tr.shape[:]
    n_params = 6

    tic = time.perf_counter()
    with device:
        if u_t_H is not None:
            ksp_navi_ngc_tr_ = compress_dictionary_chunk(ksp_navi_ngc_tr.transpose((1, 2, 3, 0)), None, u_t_H).transpose((-1, 0, 1, 2))
        else:
            ksp_navi_ngc_tr_ = ksp_navi_ngc_tr

        img_navi_pxyz = np.transpose(img_navi_xyzp, [3, 0, 1, 2])
        motion_est_gnp = np.zeros((N_GR, Nt_navi_match, n_params))
        L1_res_gn = np.zeros((N_GR, Nt_navi_match))

        for navi_ii in range(Nt_navi_match):
            ksp_gpu_gct = ksp_navi_ngc_tr_[navi_ii, ...]
        
            if match_mode == 'DictMatch':
                if not match_difference:
                    dict_gpu_mct = sp.to_device(dict_ksp_mctq[..., navi_ii], device)

                if match_difference:
                    dict_zero = dict_ksp_mctq[[zero_mo_ind], ..., navi_ii]
                    dict_gpu_mct = sp.to_device(np.delete(dict_ksp_mctq[..., navi_ii], (zero_mo_ind), axis=0), device)
                    dict_gpu_mct = dict_gpu_mct - sp.to_device(dict_zero, device)
                    dict_mo_pars_mp = np.delete(dict_mo_pars_mp, (zero_mo_ind), axis=0)
                
                if smooth_match and navi_ii > 0:
                    mo_prev_gp = motion_est_gnp[:, navi_ii-1, :]

                results_matching_dot_svd = analyze_match_interscan(ksp_gpu_gct, dict_gpu_mct, dict_mo_pars_mp, \
                                                              np.arange(N_GR), img_navi_pxyz, plot=verbose, verbose=verbose, loss="dot_cmpl",\
                                                              motion_GT=mo_GT, chunksize=dict_ksp_mctq.shape[0]//2,\
                                                              return_matched=False, mo_prev_gp=mo_prev_gp, \
                                                                   lamda_kalman=lamda_kalman, n_minima=n_minima)
            elif match_mode == 'MLP':
                results_matching_dot_svd = analyze_nn_regressor_interscan(ksp_gpu_gct, nn_model_name, motion_GT=mo_GT, \
                                               img_vols_gxyz=img_navi_pxyz, torch_device_idx=device.id)

            if navi_ii == 0 and mo_GT is None:
                mo_GT = results_matching_dot_svd['motion_GT']
                img_navi_pxyz = None

            motion_est_gnp[:, navi_ii, :] = results_matching_dot_svd['motion_Est']
            if match_mode == 'DictMatch':
                L1_res_gn[:, navi_ii] = np.squeeze(results_matching_dot_svd['L1_res'])

    res_matching = {}
    res_matching['motion_Est'] = np.squeeze(motion_est_gnp)
    res_matching['L1_res'] = L1_res_gn
    res_matching['motion_GT'] = np.copy(mo_GT)
    
    if Nt_navi_match > 1:
        # mask navi 0 and 7 for plotting as signal is low
        res_matching['motion_Est'][:, 0, :] = np.copy(res_matching['motion_Est'][:, 1, :])
        res_matching['motion_Est'][:, 7, :] = 0.5 * (res_matching['motion_Est'][:, 6, :] + res_matching['motion_Est'][:, 8, :]) 

        res_matching['motion_Est'] = res_matching['motion_Est'].reshape((-1, n_params))
        mo_GT = np.tile(res_matching['motion_GT'][:, None, :], (1, Nt_navi_match, 1))
        res_matching['motion_GT'] = mo_GT.reshape((-1, n_params))
        res_matching['L1_res'] = res_matching['L1_res'].reshape((-1, 1))

    toc = time.perf_counter()
    if verbose:
        print(f"time = {toc - tic:0.4f} seconds")  

    find_closest_in_dict(res_matching, input_gendict)
    if save_match:
        dict_name_to_save = dir_matching+f'resMatch_{save_tag}.pkl'
        with open(dict_name_to_save, 'wb') as f:
            pickle.dump(res_matching, f)
        print(f'Dict matching results are saved in {dict_name_to_save}')

    if Nt_navi_match == 1 and verbose:
        plot_moEstim(res_matching, ylabel_="Poses", save=save_match, \
                 figname=dir_figure+f'resMatch_table_{save_tag}')

#     figname_hist = dir_figure+f"resMatch_hist_{save_tag}"
#     plot_two_hist(res_matching, img_resolution=4, max_dx=5, max_dphi=15,  \
#                             figname=figname_hist, \
#                             save=save_match)

    if Nt_navi_match > 1:
        figname_traj = dir_figure+f"resMatch_motion_{save_tag}"
        plot_motion_traj(res_matching, img_resolution=4, N_GR=N_GR, Nt_navi=Nt_navi_match, save=save_match, figname=figname_traj)
        
        if smoothfit_motion_noGT:
            res_matching_withfit = fit_smooth(res_matching, 1, 0.02, through_GT=False, debug=False, \
                                              save=True, dict_name=dict_name_to_save, order=1)

        if smoothfit_motion_with_GT:
            res_matching_withfit = fit_smooth(res_matching, 1, 2e-3, through_GT=True, debug=False, \
                                              save=True, dict_name=dict_name_to_save, order=1)

            plot_motion_traj(res_matching_withfit, img_resolution=4, N_GR=16, Nt_navi=Nt_navi_match, save=save_match, \
                             figname=figname_traj+'_smooth_dict', plot_GT=False, \
                             plot_smooth_GT=True, plot_smooth_noGT=True)
            
        if smoothfit_motion_GT:
            res_matching_withfit = fit_smooth(res_matching, 1, 2e-3, through_GT=False, debug=False, motion_to_fit='motion_GT', \
                                              save=True, dict_name=dict_name_to_save, order=1)

            plot_motion_traj(res_matching_withfit, img_resolution=4, N_GR=16, Nt_navi=Nt_navi_match, save=save_match, \
                             figname=figname_traj+'_smooth', plot_Est=False, \
                             plot_smooth_GT=False, plot_smooth_noGT=False, plot_GT_smooth=True)

              
    return res_matching


saveDIR = '/local_mount/space/ladyyy/data/users/aizadan/MoEstim_Nav/Mo_Dict/MRF_231013/'
def compress_dictionary_chunk(dict_ksp_mctq, Phi_qnr, u_t_H):
    """
    Compression basis u_t_H can be ct(t') - coils are treated independently OR (ct)(ct') - coils are merged before compression.
    Not suitable for coil com
    """
    device = sp.get_device(u_t_H)
    xp = device.xp
    
    n_mo, nc, nt, nq = dict_ksp_mctq.shape
    if Phi_qnr is not None:
        n_navi, n_spir = Phi_qnr.shape[1:]
    else:
        n_navi = 1
        n_spir = 1
        
    if u_t_H is not None:
        nc_compr = u_t_H.shape[0]
    else:
        nc_compr = nc
        
    with device:        
        dict_chunk_gpu_mctrq = dict_ksp_mctq.reshape((n_mo, nc, nt // n_spir, n_spir, nq))
        del dict_ksp_mctq

        # unroll contrast subspace and compress
        if Phi_qnr is not None:
            dict_chunk_gpu_mtrcn = xp.matmul(dict_chunk_gpu_mctrq.transpose((0, 2, 3, 1, 4)), Phi_qnr.transpose((2, 0, 1))[None, None, ...])
            dict_chunk_gpu_mctn = dict_chunk_gpu_mtrcn.transpose((0, 3, 1, 2, 4)).reshape((n_mo, nc_compr, -1, n_navi))
            del dict_chunk_gpu_mtrcn
        else:
            dict_chunk_gpu_mctn = dict_chunk_gpu_mctrq[..., 0].reshape((n_mo, nc_compr, -1, n_navi))     
        del dict_chunk_gpu_mctrq
        
        if u_t_H is not None:
            dict_compr_gpu_mctn = xp.zeros((n_mo, nc_compr, u_t_H.shape[-1], n_navi), dtype=np.complex64)
            for ch_ii in range(nc_compr):
                dict_compr_gpu_mctn[:, ch_ii, ...] = xp.matmul(u_t_H[ch_ii].transpose((1, 0))[None], dict_chunk_gpu_mctn[:, ch_ii]) 
        else:
            dict_compr_gpu_mctn = dict_chunk_gpu_mctn

    return dict_compr_gpu_mctn

saveDIR = '/local_mount/space/ladyyy/data/users/aizadan/MoEstim_Nav/Mo_Dict/MRF_231013/'
def compress_memmap_dictionary(full_dict_name, full_dict_shape, u_t_H, compr_dict_name, dictionary_type='scout', \
                        Nt_navi_compr=1, Nt_navi_full=10, Phi_qt=None, chunksize=256*4, n_spirals=3):
    """
    dictionary_type : str
        scout or qscout
    Nt_navi_full : int
        number of navigator in one group
    """
    device = sp.get_device(u_t_H)
    xp = device.xp
    
    sizeof_cmpl = 8
    N_mo, N_ch, N_t, N_coef = full_dict_shape[:]
    N_ch_compr, _, Nt_compr = u_t_H.shape[:]
        
    print('Starting the dictionary compression')
    compr_dict_fullname = saveDIR + f'{compr_dict_name}' 
    dict_ksp_compr_mctn = np.memmap(compr_dict_fullname+'.dat', dtype=np.complex64, 
                      mode='w+', shape=(N_mo, N_ch_compr, Nt_compr, Nt_navi_compr))
    
    N_chunks = (N_mo + chunksize - 1) // chunksize 
    print(f'Number of data chunks {N_chunks}')    
    tr_ids_navi = np.arange(Nt_navi_full - 1, Nt_navi_full - 1 - Nt_navi_compr, -1)    
    
    tic = time.perf_counter()
    with device:
        for chunk_ii in range(N_chunks): 
            
            start = chunk_ii * chunksize
            end = min((chunk_ii + 1) * chunksize, N_mo)
            
            if dictionary_type == 'qscout':
                dict_ksp_mctq = np.zeros((end - start, N_ch, N_t, N_coef), dtype=np.complex64)
                for coef_ii in range(N_coef):
                    full_dict_fullname = saveDIR + f'{full_dict_name}_coef{coef_ii+1}'
                    byte_offset = start * N_ch * N_t * sizeof_cmpl
                    dict_ksp_mctq[..., coef_ii] = np.memmap(full_dict_fullname+'.dat', \
                                                              dtype=np.complex64, 
                                  mode='r', shape=(end - start, N_ch, N_t), offset=byte_offset)
            elif dictionary_type in ['qscout_joint', 'scout']:
                full_dict_fullname = saveDIR + f'{full_dict_name}'
                byte_offset = start * N_ch * N_t * N_coef * sizeof_cmpl
                dict_ksp_mctq = np.memmap(full_dict_fullname+'.dat', dtype=np.complex64, \
                              mode='r', shape=(end - start, N_ch, N_t, N_coef), offset=byte_offset)
            
            dict_chunk_gpu_mctrq = sp.to_device(dict_ksp_mctq, device).reshape((end - start, N_ch, N_t//3, 3, N_coef))
            del dict_ksp_mctq
            
            # unroll contrast subspace and compress
            for TR_navi_ii, TR_navi in enumerate(tr_ids_navi):
                
                if Phi_qt is not None:
                    Phi_rq_gpu = sp.to_device(Phi_qt[:, 3 * TR_navi: 3 * (TR_navi + 1)], device).T
                    
                    dict_chunk_gpu_mctr = xp.matmul(dict_chunk_gpu_mctrq[..., None, :], Phi_rq_gpu[:, :, None])[..., 0, 0]
                else:
                    dict_chunk_gpu_mctr = dict_chunk_gpu_mctrq[..., 0]     
                    
                dict_chunk_gpu_mctr = dict_chunk_gpu_mctr.reshape((end - start, N_ch_compr, -1))
                dict_chunk_gpu_mct = xp.matmul(dict_chunk_gpu_mctr, u_t_H[0])  
                nii = TR_navi_ii if Nt_navi_compr == 1 else TR_navi
                dict_ksp_compr_mctn[start:end, :, :, nii] = dict_chunk_gpu_mct.get()
                
    dict_ksp_compr_mctn.flush()
    toc = time.perf_counter()
    print(f"Compression time = {toc - tic:0.4f} seconds")    


def plot_2signals_phase(ksp_nav, y_sim_trueMo, Nt_lo=0, Nt_up=0, ch=0, labels=[]):
    if Nt_up==0:
        Nt_up = ksp_nav.shape[1]
    plt.figure()
    plt.plot(np.arange(Nt_lo,Nt_up), np.unwrap(np.angle(ksp_nav)[ch,:]),"-o", label=labels[0])
    plt.plot(np.arange(Nt_lo,Nt_up), np.unwrap(np.angle(y_sim_trueMo)[ch,:]),"-o", label=labels[1])
    plt.legend()
    plt.ylabel("Signal")
    plt.xlabel("t")
    plt.title("Signal phases")
    plt.tight_layout()
    

def plot_2signals(ksp_nav, y_sim_trueMo, Nt_lo=0, Nt_up=0, ch=0, labels=[]):

    if Nt_up==0:
        Nt_up = ksp_nav.shape[1]
    plt.figure()
    plt.plot(np.arange(Nt_lo,Nt_up), (np.abs(ksp_nav))[ch,Nt_lo:Nt_up],"-o", label=labels[0])
    plt.plot(np.arange(Nt_lo,Nt_up), (np.abs(y_sim_trueMo))[ch,Nt_lo:Nt_up],"-o", label=labels[1])
    plt.legend()
    plt.ylabel("Signal")
    plt.xlabel("t")
    plt.tight_layout()
    
def l2_cmpl(in_arr):
    return np.sqrt( np.sum( (np.abs(in_arr))**2 ) )

def func_norm(arr):
    eps = 1e-8
    norm_factor = np.linalg.norm((arr)) + eps
#     norm_factor = np.percentile(np.abs(arr), 95)
    return arr / norm_factor

def func_norm_std(arr):
    eps = 1e-8
    abs_vec = np.abs(arr)
    p5, p95 = np.percentile(abs_vec, [5,95])
    arr_std = (arr-p5)/(p95-p5)
    return arr_std

def dot_prod_cmpl(x, y):
    """
    Dot product of the two complex vectors
    """
    eps = 1e-8
    return np.vdot(x, y) / (np.linalg.norm(x)+eps)*(np.linalg.norm(y)+eps)

def dot_prod(x, y):
    """
    Dot product of the two complex vectors split into channels
    """
    x_re = np.concatenate((x.real, x.imag), axis=1)
    y_re = np.concatenate((y.real, y.imag), axis=1)
    return np.vdot(x_re, y_re) / (np.linalg.norm(x_re)*np.linalg.norm(y_re))


def gEVD_dictionary(dict_ksp_nmct, ksp_stat_ngct=None, xp=np, method='gEVD', unit_variance=False, demean_data=True, scale_variance=True):
    """
    Expected normalized arrays
    """
    device = sp.get_device(dict_ksp_nmct)
    with device:
        if unit_variance and demean_data:
            dict_ksp_nmct_demean = (dict_ksp_nmct - xp.mean(dict_ksp_nmct, axis=1, keepdims=True)) / (1e-2 + xp.std(dict_ksp_nmct, axis=1, keepdims=True))
        elif demean_data:
            dict_ksp_nmct_demean = dict_ksp_nmct - xp.mean(dict_ksp_nmct, axis=1, keepdims=True)   
        else:
            dict_ksp_nmct_demean = dict_ksp_nmct
            
        ng_dict = dict_ksp_nmct_demean.shape[0] * dict_ksp_nmct_demean.shape[1]
        D_nm_ct = dict_ksp_nmct_demean.reshape((ng_dict, -1))
        del dict_ksp_nmct_demean, dict_ksp_nmct
        
        Q = xp.matmul(xp.conj(D_nm_ct).T, D_nm_ct) 
        if scale_variance: 
            Q /= (ng_dict - 1)
        print(f'Q shape {Q.shape}')
        
        if method in ['gEVD', 'pinv_SVD']:
            if unit_variance and demean_data:
                ksp_stat_ngct_demean = (ksp_stat_ngct - xp.mean(ksp_stat_ngct, axis=1, keepdims=True)) / ( 1e-2 + xp.std(ksp_stat_ngct, axis=1, keepdims=True))
            elif demean_data:
                ksp_stat_ngct_demean = ksp_stat_ngct - xp.mean(ksp_stat_ngct, axis=1, keepdims=True)   
            else:
                ksp_stat_ngct_demean = ksp_stat_ngct
                
            ng_ksp_stat = ksp_stat_ngct_demean.shape[0] * ksp_stat_ngct_demean.shape[1]
            s_ng_ct = ksp_stat_ngct_demean.reshape((ng_ksp_stat, -1))
            del ksp_stat_ngct_demean, ksp_stat_ngct
            
            P = xp.matmul(xp.conj(s_ng_ct).T, s_ng_ct)
            if scale_variance:
                P /= (ng_ksp_stat - 1)
            del s_ng_ct
            print(f'P shape {P.shape}')
            print(f'max values of P and Q', xp.max(xp.abs(P)), xp.max(xp.abs(Q)))
            
            if  method == 'gEVD':
                result = eig(Q.get(), P.get(), overwrite_a=True, overwrite_b=True) # right e.v.
                if_symmetric = False
            elif method == 'pinv_SVD':
                P_inv_Q = xp.matmul(xp.linalg.pinv(P), Q)
                del P, Q
                if_symmetric = xp.allclose(P_inv_Q, xp.conj(P_inv_Q).T, rtol=1e-6, atol=1e-8)
                result = xp.linalg.svd(P_inv_Q, full_matrices=False)
        
        if method == 'SVD':
            result = xp.linalg.svd(Q, full_matrices=False)
            if_symmetric = False 
                
    return result, if_symmetric