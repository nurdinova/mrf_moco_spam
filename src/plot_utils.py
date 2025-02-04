
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def find_closest_in_dict(results_matching, input_gendict):
    
    """
    find entries in the dict closest to motion_GT, calculate minimum errors in dict and 
    add 'motion_in_dict' and 'min_errors_in_dict' entries to the results dict
    """ 
    motion_GT_afni_pm = np.copy(results_matching["motion_GT"])
    motion_pars = ["shifts_x", "shifts_y", "shifts_z", "angles_x", "angles_y", "angles_z"]

    motion_closest_in_dict = np.zeros_like(motion_GT_afni_pm)
    min_errors_in_dict = np.zeros_like(motion_GT_afni_pm)
    for mo_par_ii, mo_par in enumerate(motion_pars):
        mo_error = np.abs(motion_GT_afni_pm[:,mo_par_ii,np.newaxis].T - input_gendict[mo_par][:,np.newaxis])
        min_pars = np.argmin(mo_error, axis=0)
        motion_closest_in_dict[:,mo_par_ii] = input_gendict[mo_par][[min_pars]]
        min_errors_in_dict[:,mo_par_ii] = motion_GT_afni_pm[:,mo_par_ii] - motion_closest_in_dict[:,mo_par_ii]

    results_matching["motion_in_dict"] = motion_closest_in_dict
    results_matching["min_errors_in_dict"] = -min_errors_in_dict
    return results_matching

def plot_two_hist(res_match, img_resolution=4, max_dx=1, max_dphi=1,  \
                        figname='', titles=['rotation', 'translation'], units=['degrees', 'mm'], \
                        save=False):
    
    err_transl_GT_estim = img_resolution * (res_match['motion_Est'][:, :3].flatten() - \
                                            res_match['motion_GT'][:, :3].flatten())
    err_transl_GT_dict = img_resolution * res_match['min_errors_in_dict'][:, :3].flatten()
    
    transl_res = {r'$\Delta_{Match} - \Delta_{GT}$, mm': err_transl_GT_estim, \
             r'$\Delta_{GT} - \Delta_{Grid}$, mm': err_transl_GT_dict}
    df_transl = pd.DataFrame(transl_res)

    
    err_GT_estim = 180/np.pi*(res_match['motion_Est'][:, 3:].flatten() - res_match['motion_GT'][:, 3:].flatten()) \
                                                
    err_GT_dict = 180/np.pi*res_match['min_errors_in_dict'][:,3:].flatten()
    
    rotat_res = {r'$\theta_{Match} - \theta_{GT}$, degrees': err_GT_estim, \
                 r'$\theta_{GT} - \theta_{Grid}$, degrees': err_GT_dict}
    df_rotat = pd.DataFrame(rotat_res)

    max_values = np.array([max_dphi, max_dx])
    dfs = [df_rotat, df_transl]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for df_ii, df in enumerate(dfs):
        for row_ii in range(2):
            sns.histplot(data=df, x=list(df.columns)[row_ii], stat='count', bins=21, \
                         binrange=(-max_values[df_ii], max_values[df_ii]), \
                        ax=axs[row_ii, df_ii])
#         plt.tight_layout()
        if save:
            plt.savefig(figname, dpi=300, bbox_inches='tight', \
               transparent=True, pad_inches=0)
    


def plot_motion_traj(res_match_dict, img_resolution=4, N_GR=16, Nt_navi=10, save=False, figname='', plot_Est=True, plot_GT=True, \
                     plot_smooth_GT=False, plot_smooth_noGT=False, plot_GT_smooth=False, figsize=(10, 5), \
                     translation_lims=[-7, 7], rotation_lims=[-20, 20]):
    """
    res_match_dict : dictionary
        Matching results dictionary with the fields 'motion_Est', 'motion_GT' of shape (N_mo, 6)
    """
    N_TR_navi = Nt_navi * N_GR
    n_params = 6
    time = (47 + np.arange(N_TR_navi) * 50) * 12.5e-3 # s
    
    mo_multiplier = np.array([img_resolution, img_resolution, img_resolution, \
                              180 / np.pi, 180 / np.pi, 180 / np.pi]).reshape((1, 6))
    
    if plot_Est:
        motion_rp = np.copy(res_match_dict['motion_Est']) * mo_multiplier
    if plot_GT:
        motion_imreg_gp = np.copy(res_match_dict['motion_GT']) * mo_multiplier # [9::Nt_navi, :]
    if plot_smooth_GT:
        motion_fit = np.copy(res_match_dict['motion_smooth_GT']) * mo_multiplier
    if plot_smooth_noGT:
        motion_fit_noGT = np.copy(res_match_dict['motion_smooth_noGT']) * mo_multiplier
    if plot_GT_smooth:
        motion_GT_fit = np.copy(res_match_dict['motion_GT_smooth']) * mo_multiplier
        
    labels = [r'$\Delta x$, mm', r'$\Delta y$, mm', r'$\Delta z$, mm', r'$\theta_{x}$, degrees', \
              r'$\theta_{y}$, degrees', r'$\theta_{z}$, degrees']
    
    n_cols = 3
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    for par in range(6):
        if plot_Est:
            axs[par//n_cols, par%n_cols].plot(time, motion_rp[:, par], '--o', label='DM')
        if plot_GT:
            axs[par//n_cols, par%n_cols].plot(time[Nt_navi-1::Nt_navi], motion_imreg_gp[Nt_navi-1::Nt_navi, par], '-o', label='GT')
        if plot_smooth_GT:
            axs[par//n_cols, par%n_cols].plot(time, motion_fit[:, par], '-', linewidth=2, label='DM, smooth fit using GT')
        if plot_smooth_noGT:
            axs[par//n_cols, par%n_cols].plot(time, motion_fit_noGT[:, par], '-', linewidth=2, label='DM, smooth fit')
        if plot_GT_smooth:
            axs[par//n_cols, par%n_cols].plot(time, motion_GT_fit[:, par], '-', linewidth=2, label='GT, smooth fit')

        axs[par//n_cols, par%n_cols].set_xlabel('Time, s')
        axs[par//n_cols, par%n_cols].set_title(labels[par])
        axs[par//n_cols, par%n_cols].legend()
        if  par < 3:
            axs[par//n_cols, par%n_cols].set_ylim(translation_lims)
        elif par >= 3:
            axs[par//n_cols, par%n_cols].set_ylim(rotation_lims)
    plt.tight_layout()
    if save:
        plt.savefig(figname, dpi=300, bbox_inches='tight', \
               transparent=True, pad_inches=0)
        
def plot_3dimg(img,N):
    fig,axs = plt.subplots(1,3, figsize=(6,3))
    axs[0].imshow(np.abs(img[:,:,N//2]))
    axs[0].set_title("xy-plane")
    axs[0].axis("off")
    
    axs[1].imshow(np.abs(img[N//2,:,:]))
    axs[1].set_title("yz-plane")
    axs[1].axis("off")
    
    axs[2].imshow(np.abs(img[:,N//2,:]))
    axs[2].set_title("xz-plane")
    axs[2].axis("off")
    
    
def plot_imgs_N_diff(im1, im2, N, diff_scale=1, vmax_factor=None, labels=['Motion', 'Reference'], save=False, save_name=''):
                
    if vmax_factor is None:
        vmax_factor = [1, 0.6, 0.5]
    vmax = [1., 1., 1]
    
    names = ['_mov', '_ref']
    for img_ii, img_ in enumerate([im1, im2]):
        fig, axs = plt.subplots(1,3, figsize=(9,4))
        
        img = np.copy(img_)
        im = np.abs(img[:,::-1,N//2].T)
        im /= np.max(im)
        axs[0].imshow(im, vmax=vmax[0]**vmax_factor[0])
        axs[0].set_title("ax")
        axs[0].axis("off")

        im = np.abs(img[N//2,:,::-1].T)
        im /= np.max(im)
        axs[1].imshow(im, vmax=vmax[1]*vmax_factor[1])
        axs[1].set_title("sag")
        axs[1].axis("off")

        im = np.abs(img[:,N//2,::-1].T)
        im /= np.max(im)
        axs[2].imshow(im, vmax=vmax[2]*vmax_factor[2])
        axs[2].set_title("cor")
        axs[2].axis("off")
        plt.suptitle(labels[img_ii])
        if save:
            plt.savefig(save_name+names[img_ii])

    
    fig,axs = plt.subplots(1,3, figsize=(9,4))
    im1_norm = np.copy(im1[:,::-1,N//2].T)
    im1_norm /= np.max(np.abs(im1_norm))
    im2_norm = np.copy(im2[:,::-1,N//2].T)
    im2_norm /= np.max(np.abs(im2_norm))
    axs[0].imshow(np.abs(im1_norm-im2_norm), vmax=vmax[0]*vmax_factor[0]/diff_scale)
    axs[0].set_title("ax")
    axs[0].axis("off")

    im1_norm = np.copy(im1[N//2,:,::-1].T)
    im1_norm /= np.max(np.abs(im1_norm))
    im2_norm = np.copy(im2[N//2,:,::-1].T)
    im2_norm /= np.max(np.abs(im2_norm))
    axs[1].imshow(np.abs(im1_norm-im2_norm), vmax=vmax[1]*vmax_factor[1]/diff_scale)
    axs[1].set_title("sag")
    axs[1].axis("off")

    im1_norm = np.copy(im1[:,N//2,::-1].T)
    im1_norm /= np.max(np.abs(im1_norm))
    im2_norm = np.copy(im2[:,N//2,::-1].T)
    im2_norm /= np.max(np.abs(im2_norm))
    axs[2].imshow(np.abs(im1_norm-im2_norm), vmax=vmax[2]*vmax_factor[2]/diff_scale)
    axs[2].set_title("cor")
    axs[2].axis("off")
    plt.suptitle(labels[0]+' - '+labels[1])
    if save: 
        plt.savefig(save_name+'_diffx'+str(diff_scale))
   