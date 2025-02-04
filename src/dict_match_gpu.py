import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import sigpy as sp
import time


def matching_chunk_gpu(dict_mct_chunk, ksp_nav_ct, loss, xp, ksp_dict_zero=None, ksp_nav_zero=None, n_minima=1):
    
    # Code for GPU dictionary matching on the chunk
    Nm, Nc, Nt = dict_mct_chunk.shape
    device = sp.get_device(dict_mct_chunk)
    with device:
        if (loss == "l1"):
            loss_m = xp.sum(xp.abs((dict_mct_chunk-ksp_nav_ct)), axis=(1,2))
        elif (loss=="l2"):
            loss_m = xp.sqrt(xp.sum(xp.abs((dict_mct_chunk-ksp_nav_ct))**2, axis=(1,2)))
        elif loss == "dot":
            dict_mct_re_im = xp.concatenate((dict_mct_chunk.real, dict_mct_chunk.imag), axis=-1)
            ksp_nav_ct_re_im = xp.concatenate((ksp_nav_ct.real, ksp_nav_ct.imag), axis=-1)
            N_mo = dict_mct_re_im.shape[0]
            loss_m = xp.zeros((N_mo, 1))
            for mo_ii in range(N_mo):
                loss_m[mo_ii,0] = xp.vdot(dict_mct_re_im[mo_ii], ksp_nav_ct_re_im) / (xp.linalg.norm(dict_mct_re_im[mo_ii]) * xp.linalg.norm(ksp_nav_ct_re_im))
        elif loss == "dot_cmpl":
            loss_m = calc_dot_cmpl(dict_mct_chunk, ksp_nav_ct, xp)

        if loss in ["l1", "l2"]:
            mo_idx_min_chunk = xp.argsort(loss_m)[:n_minima]
        elif loss in ["dot", "dot_cmpl"]:
            mo_idx_min_chunk = xp.argsort(loss_m)[::-1][:n_minima]    
    return xp.squeeze(loss_m), mo_idx_min_chunk

def calc_dot_cmpl(dict_mct_chunk, ksp_nav_ct, xp):
    N_mo = dict_mct_chunk.shape[0]
    device = sp.get_device(ksp_nav_ct)
    with device:
        dot_prod = xp.abs(xp.dot(xp.conj(dict_mct_chunk.reshape((N_mo, -1))), ksp_nav_ct.reshape((-1, )))) \
            / (xp.linalg.norm(dict_mct_chunk, axis=(-1,-2)) * xp.linalg.norm(ksp_nav_ct))
    return dot_prod

def match_toDict_gpu(ksp_nav_ct, dict_mct, dict_mo_idx_mp, motion_true=None, Nt_lo=0, Nt_up=0, loss="l1", plot=False, \
                     weighting=0, sens_maps=None, crds_nav=None, subspace_u=None, verbose=0, device=None, chunksize=None, \
                     Phi=None, T_ind=None, ksp_dict_zero=None, ksp_nav_zero=None, n_minima_chunk=1, mo_prev=None, lamda_kalman=0.):
    """
    Matching provided k-space navigator signal to the motion dictionary.
    
    N_motion_parameters - meant in the dictionary.
    Input:
        ksp_nav_ct : array (N_channels, N_timepoints)
            new normalized navigator signal
        dict_mct : array or np.memmap of size (N_motion_parameters, N_channels, N_timepoints)
            normalized in the same way as the new navigator signal
        dict_mo_idx_mp : array (N_motion_parameters, N_motion_dof=6)
            map of dictionary indices and motion parameters
        motion_true : array of size (N_motion_dof=6)
            if provided, the result of matching will be compared to it
        Nt_lo, Nt_up : lower and upper time boundaries for the provided signals
        loss : string
            options: "l1", "l2", "dot", "dot_cmpl"   
            loss function for dictionary matching 
        plot : bool
            if plot the matched signal and input
        weighting : string
            in process 
            weighting of the matching loss function
        subspace : bool
            if the provided dictionary is in lower-dimensional space
        verbose: 
            allow prints of the results
        
    Return:
        ksp_matched : array of size (N_channels, N_timepoints)
        ksp_loss : array of size (N_motion_parameters)
            loss on each dictionary entry
        motion_matched : array of size (N_motion_dof)
            matched motion parameters
    
    """
    device = sp.get_device(ksp_nav_ct)
    mvd = lambda x : sp.to_device(x, device)
    xp = device.xp
    
    if chunksize is None:
        chunksize = 256*64
        
    with device:
        subspace = False
        if subspace_u is not None:
            subspace = True
            u_H_T = xp.conj(subspace_u)

        if verbose:
            tic = time.perf_counter()
    
        # if time boundaries are provided, chop the signals
        if Nt_up == 0:
            Nt_up = ksp_nav_ct.shape[1]
        ksp_nav_ct_gpu = ksp_nav_ct[:, Nt_lo:Nt_up]

        n_mo_pars, N_ch, N_t = dict_mct.shape[:3]

        N_chunks = (n_mo_pars + chunksize - 1) // chunksize        
        ksp_loss_m_gpu = xp.zeros((n_mo_pars), dtype=xp.float64)
        min_idx_chunks = xp.zeros((N_chunks, n_minima_chunk), dtype=np.int16) # motion index with the lowest loss

        for chunk_ii in range(N_chunks):

            start = chunk_ii * chunksize
            end = min((chunk_ii + 1) * chunksize, n_mo_pars)

            dict_chunk_gpu = dict_mct[start:end]
            if Phi is not None:
                Phi_gpu = mvd(Phi[....T_ind])
                dict_chunk_gpu = xp.matmul(dict_chunk_gpu, Phi_gpu)
            if subspace:
                dict_chunk_gpu = xp.matmul(dict_chunk_gpu, u_H_T)

            dict_mct_chunk_gpu = dict_chunk_gpu[:, :, Nt_lo:Nt_up]            
            
            if verbose:
                toc = time.perf_counter()
                print(f"Prepping the data time = {toc - tic:0.4f} seconds")
                tic = time.perf_counter()
                
            ksp_loss_m_gpu[start:end], min_idx_ = matching_chunk_gpu(dict_mct_chunk_gpu, ksp_nav_ct_gpu, loss, xp, ksp_dict_zero, ksp_nav_zero, n_minima=n_minima_chunk)
            
            if verbose:
                toc = time.perf_counter()
                print(f"Pure matching time = {toc - tic:0.4f} seconds")
            
            min_idx_chunks[chunk_ii] = min_idx_ + start
            del dict_mct_chunk_gpu
            xp._default_memory_pool.free_all_blocks()

        ksp_loss_m_cpu = ksp_loss_m_gpu.get()
        if (verbose > 0):
            plt.figure()
            plt.plot(ksp_loss_m_cpu)
            plt.title(loss+" loss")
            plt.xlabel("Dict index")
            plt.tight_layout()

        with device:
            # global optimum
            min_idx_chunks = min_idx_chunks.ravel().get()
            if mo_prev is None:
                loss_total = xp.zeros((len(min_idx_chunks), ))
            else:
                loss_total = lamda_kalman * xp.linalg.norm(mvd(mo_prev - dict_mo_idx_mp[min_idx_chunks]), axis=-1)
            if loss in ["l1", "l2", "corr_diff"]:
                loss_total += ksp_loss_m_gpu[min_idx_chunks]
            elif loss in ["dot", "dot_cmpl"]:
                loss_total -= ksp_loss_m_gpu[min_idx_chunks]
            mo_idx_min = int(min_idx_chunks[int(xp.argmin(loss_total))])

    motion_matched = dict_mo_idx_mp[mo_idx_min,:]
    ksp_matched_ct = dict_mct[mo_idx_min].get()
    if Phi is not None:
        ksp_matched_ct = np.matmul(ksp_matched_ct, Phi[....T_ind])
    if subspace:
        ksp_matched_ct = np.matmul(ksp_matched_ct, np.conj(subspace_u))

    if (verbose > 0):
#         np.set_printoptions(precision=2)
        print("Estimated motion params = ", motion_matched, "mm/rads" )
        if motion_true is not None:
            print("True motion params = \t ", motion_true, "mm/rads" )
            print("Abs Differences = \t ", (np.abs(motion_true-motion_matched)))
            print("Sum of rel errors = \t %.2f" % np.sum(np.abs(motion_true - motion_matched) /                                                          np.abs(motion_true)))
            print("Weighted sum of rel errors = \t %.2f" % np.sum(np.abs(motion_true - motion_matched)))

    if plot:
        plot_2signals(ksp_nav_ct.get(), ksp_matched_ct, Nt_lo=0, Nt_up=0, ch=0, labels=["Input", "Matched"])
        plot_2signals_phase(ksp_nav_ct.get(), ksp_matched_ct, Nt_lo=0, Nt_up=0, ch=0, labels=["Input", "Matched"])
    return ksp_matched_ct, ksp_loss_m_cpu, (motion_matched) 
