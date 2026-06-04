#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ## Motion Estimation 
# ### from Spiral Navigators inserted every ~40 TR within a 3D SPI MRF sequence
#
# Short spiral navigators (3 repetitions, ~12 ms each) are interleaved with MRF acquisition groups of 500 TR. Each navigator group samples the same k-space trajectory, making its signal sensitive to rigid-body head motion relative to a Quantitative-scout (Q-scout) reference image.
#
# ---
#
# **Pipeline**
#
# | Step | Description |
# |------|-------------|
# | 1. Data loading from cfl-files | Navigator k-space, coil sensitivities, trajectory and image navigators across series. Optional coil compression. |
# | 2. Static calibration | Per-channel baseline from a static scan; used to match simulated and in-vivo signal scales (gEVD / pinv-SVD). |
# | 3. Subspace projection | Q-scout coefficients are projected into the MRF temporal subspace → per-contrast input for motion-dictionary simulation. |
# | 4. Motion SVD basis | Small motion dictionary (±1 vox / ±0.5°) decomposed into a per-channel time-compression basis **U**. |
# | 5. Dictionary generation | Fine 6-DOF grid simulated per navigator contrast, stored as memory-mapped files compressed through **U**. |
# | 6. Dictionary matching | Measured navigators projected through **U**, matched via complex dot-product. Optional Kalman regularisation. |
# | 7. Evaluation | Estimated trajectories vs. image nav registration-based ground-truth; correlation and MAD reported. Results saved for further motion optimization and reconstruction. |
#

# %%
import argparse
import json
import os
import pickle
import sys
import time


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sigpy as sp
import sigpy.plot as pl
from scipy.io import savemat

sys.path.append("/usr/local/app/bart/bart-0.6.00/python/")
sys.path.append("/local_mount/space/mayday/data/users/aizadan/dictgator/")

import cfl
from src.utils import clean_gpu, norm_min_max_zero, norm_signals_to_real, signal_power, apply_svd_basis
from src.analysis.metrics import compute_motion_metrics
from src.Forward_NavAcq_2 import call_afni
from src.dict_gen_gpu import dict_gen
from src.analysis.plot_dictMatch import plot_3dimg, plot_2signals, dot_sim
from src.analysis.plot_motion import plot_motion_traj
from src.simulate_Nav_signal import load_data
from src.dict_match import run_matching
from src.simulate_dB0 import gEVD_dictionary

# %% [markdown]
# ### Figure Style

# %%
_font = 12
plt.rcParams.update({
    'axes.linewidth': 3, 'font.size': _font, 'figure.figsize': (6, 4),
    'ytick.labelsize': _font, 'xtick.labelsize': _font,
    'font.weight': 'bold', 'axes.labelweight': 'bold',
    'axes.titleweight': 'bold', 'figure.titleweight': 'bold',
    'grid.linewidth': 2, 'image.cmap': 'gray',
})

# %% [markdown]
# ### Configuration
#
# Run as a notebook: `args` defaults are used.
# Run as a script:   `python MoEstim_Invivo_250425.py --date 250425 --gpu 2 ...`

# %%
_DESCRIPTION = (
    "Motion estimation from spiral navigators (3 arms, ~12 ms each) interleaved\n"
    "every ~40 TR within a 3D SPI MRF sequence.\n\n"
    "Pipeline: data loading → static calibration → subspace projection\n"
    "  → SVD basis (navigator signals) → dictionary generation\n"
    "  → dictionary matching → evaluation & export."
)

def _build_parser():
    p = argparse.ArgumentParser(
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--date',               default='250425', help='Scan date tag')
    p.add_argument('--dataset',            type=int, default=0, help='Dataset index in data_info.json')
    p.add_argument('--generate-dict',      action='store_true', help='Generate full motion dictionary')
    p.add_argument('--generate-calib-dict',action='store_true', help='Generate calibration dictionary for SVD basis')
    p.add_argument('--no-svd',             action='store_false', dest='do_svd',
                   help='Skip SVD time-compression of navigator signals')
    p.add_argument('--svd-method',         default='pinv_SVD', choices=['pinv_SVD', 'gEVD', 'whiten_SVD'])
    p.add_argument('--gpu',                type=int, default=2, help='CUDA device index')
    p.add_argument('--save-suffix',        default=None, help='Override auto-generated save suffix')
    p.set_defaults(do_svd=True)
    return p

args = _build_parser().parse_args([] if 'ipykernel' in sys.modules else None)

# %%
# ---- parameters (edit for interactive runs) ----
date_tag        = args.date
dataset_number  = args.dataset

generate_dictionary      = args.generate_dict
generate_calib_dictionary = args.generate_calib_dict
do_SVD_dictionary        = args.do_svd
svd_compr_method         = args.svd_method

# flags not exposed to CLI
do_coil_compr        = False
select_coils         = True
select_spirals       = True
coils_to_select      = [0, 1, 2, 3, 4]
navis_to_use         = [12]
n_params             = 6           # motion DOF
do_register_img_nav  = False
figure_save_suffix   = 'noKalman_smallDict'

# k-space / readout geometry
N_READOUT_PER_SPIRAL = 1658   # readout points per spiral arm
N_NOISE_FRAMES       = 200    # trailing frames used for noise covariance

# time window for dictionary simulation (readout samples)
Nt_lo = 0
Nt_up = N_READOUT_PER_SPIRAL

# SVD component range used during matching
Nt_new_lo = 0
Nt_new_up = 120

# motion dictionary grid resolution
DICT_STEP_DEG   = 0.5   # angular step [degrees] for both SVD-basis and full grids
DICT_STEP_PIXEL = 0.5   # translation step [voxels]
DICT_MAX_STEPS  = 1.0   # half-width of the SVD-basis grid [steps]

# output path for external pipeline export
SAVEMAT_DIR = './'

# static calibration: which acquisition series / groups serve as the static in-vivo reference
# calibration is needed for gEVD / pinv_SVD / whiten_SVD; not for plain SVD
calibrate_wrt_static = svd_compr_method in ('gEVD', 'pinv_SVD', 'whiten_SVD')
CALIB_SERIES_IDX     = 0    # index into dir_diff_poses: series that holds the static scan
CALIB_GROUPS         = [0]  # group indices within that series used as the static baseline

# %% [markdown]
# ### Device & Paths

# %%
device = sp.Device(args.gpu)
xp     = device.xp
mvc    = lambda x: sp.to_device(x, sp.cpu_device)
mvd    = lambda x: sp.to_device(x, device)

# %%
DICTGATOR_DIR = '/local_mount/space/mayday/data/users/aizadan/dictgator/'
JSON_FILENAME = os.path.join(DICTGATOR_DIR, 'data_info.json')

with open(JSON_FILENAME, 'r') as fh:
    metadata = json.load(fh)[date_tag][dataset_number]

dir_matching  = metadata['dir_matching']
dir_figure    = metadata['dir_figures']
saveDIR       = metadata['saveDIR']
data_path_new = metadata['recon_data_path']
N_navi_tr     = metadata['N_navi_tr']

dir_diff_poses = [f'Series{s}/' for s in metadata['Series']]
dir_diff_poses = [dir_diff_poses[0], dir_diff_poses[4]]

save_suffix = args.save_suffix or f'final_{svd_compr_method}_DictMatch_ser4_v2'

os.makedirs(saveDIR, exist_ok=True)
os.makedirs(dir_matching, exist_ok=True)
os.makedirs(f'{data_path_new}dictgator/', exist_ok=True)
os.makedirs(f'{saveDIR}dictgator', exist_ok=True)

# %% [markdown]
# ### Data Loading

# %%
for ser_ii in range(len(dir_diff_poses)):
    data = load_data(dir_diff_poses[ser_ii], data_path=data_path_new, scout='dummy',
                     matching='interTR', coil_compr=True, n_spirals=N_navi_tr)
    if ser_ii == 0:
        _, sens_dummy_xyzc, coords_navi_c_tr, _, img_navi_end_xyzp, img_dummy2_xyzc = data
    ksp_navi = data[3][:, 2:, :5, ...]
    ksp_navi_ngc_tr = ksp_navi if ser_ii == 0 \
                      else np.concatenate((ksp_navi_ngc_tr, ksp_navi), axis=1)

print('ksp_navi_ngc_tr', ksp_navi_ngc_tr.shape)

if select_coils:
    N_ch = len(coils_to_select)
    ksp_navi_ngc_tr = ksp_navi_ngc_tr[:, :, coils_to_select]
    sens_dummy_xyzc = sens_dummy_xyzc[..., coils_to_select]

Nt_navi, N_GR, N_ch, N_t = ksp_navi_ngc_tr.shape
nx, ny, nz = img_navi_end_xyzp.shape[:3]
N_GR_scan  = N_GR // len(dir_diff_poses)

if select_spirals:
    spirals_to_select = metadata['spirals_to_select']

    coords_navi_c_tr = (
        coords_navi_c_tr.reshape((3, -1, 6))
        [..., :N_READOUT_PER_SPIRAL, spirals_to_select]
        .reshape((3, -1))
    )
    ksp_navi_ngc_tr = (
        (-1j) * ksp_navi_ngc_tr.reshape((Nt_navi, N_GR, N_ch, -1, N_navi_tr))
        [..., :N_READOUT_PER_SPIRAL, spirals_to_select]
        .reshape((Nt_navi, N_GR, N_ch, -1))
    )
    N_navi_tr = len(spirals_to_select)
    N_t       = ksp_navi_ngc_tr.shape[-1]

sens_dummy_cxyz = sens_dummy_xyzc.transpose((3, 0, 1, 2))[:N_ch]
print('sens_dummy_cxyz  ', sens_dummy_cxyz.shape)
print('coords_navi_c_tr ', coords_navi_c_tr.shape)

if calibrate_wrt_static:
    N_t_spiral = N_t // N_navi_tr

print('ksp_navi_ngct    ', ksp_navi_ngc_tr.shape)

# %% [markdown]
# ### Image Navigators & Ground-Truth Motion

# %%
if do_register_img_nav:
    N_series = len(dir_diff_poses)
    N_GR     = 36 * N_series

    img_all_series_xyzp = np.zeros((nx, ny, nz, N_GR), dtype=np.complex64)
    for ser_ii in range(N_series):
        img_navi = cfl.readcfl(data_path_new + dir_diff_poses[ser_ii] + 'recon_navi_xyzc_cc')
        img_navi /= np.max(np.abs(img_navi[..., 0]))
        img_all_series_xyzp[..., ser_ii * N_GR_scan : (ser_ii + 1) * N_GR_scan] = img_navi

    _, motion_GT = call_afni(img_all_series_xyzp, pre_saved=False, save_out=False)
    motion_GT[:, 4] *= -1  # AFNI outputs CCW; operator rotates CW

    res_match_dict = {'motion_GT': np.copy(motion_GT)}
    res_match_dict['motion_GT'][:, 3:] *= np.pi / 180

    plot_motion_traj(res_match_dict, img_resolution=1, Nt_navi=1, N_GR=N_GR,
                     plot_Est=False, translation_lims=[-1, 1], rotation_lims=[-5, 5])
    # np.save(f'{data_path_new}dictgator/motion_GT_all.npy', res_match_dict['motion_GT'])

# %%
mo_GT = np.load(f'{data_path_new}dictgator/motion_GT_all.npy')
mo_GT = np.concatenate((mo_GT[:24], mo_GT[24*4:24*5]), axis=0)
print('mo_GT', mo_GT.shape)

# %% [markdown]
# ### (Optional) Coil Compression

# %%
if do_coil_compr:
    with device:
        N_ch_svd = 5
        u_coil = mvd(cfl.readcfl(f'{data_path_new}{dir_diff_poses[0]}CC_Vc_full')[:, :N_ch_svd])

        ksp_navi_ngc_tr = mvc(
            apply_svd_basis(mvd(ksp_navi_ngc_tr).transpose((0, 1, 3, 2)), u_coil)
        ).transpose((0, 1, 3, 2))

        sens_dummy_xyzc = apply_svd_basis(mvd(sens_dummy_xyzc), u_coil).get()
        sens_dummy_cxyz = sens_dummy_xyzc.transpose((-1, 0, 1, 2))

        N_ch = N_ch_svd
        print('ksp_navi_ngc_tr', ksp_navi_ngc_tr.shape)
        print('sens_dummy_xyzc', sens_dummy_xyzc.shape)

# %% [markdown]
# ### Static Calibration Data

# %%
if calibrate_wrt_static:
    ksp_spinav_ctgn = cfl.readcfl(data_path_new + dir_diff_poses[CALIB_SERIES_IDX] + 'ksp_spinav_ctgr_cc')
    if select_coils:
        ksp_spinav_ctgn = ksp_spinav_ctgn[coils_to_select]
    elif do_coil_compr:
        with device:
            ksp_spinav_ctgn = mvc(
                apply_svd_basis(mvd(ksp_spinav_ctgn).transpose((1, 2, 3, 0)), u_coil)
            ).transpose((3, 0, 1, 2))

    ksp_spinav_ctgn = ksp_spinav_ctgn[:, :N_READOUT_PER_SPIRAL, ...]

    # drop dummy groups (first 2) then select only the calibration groups
    ksp_spinav_ctgn = ksp_spinav_ctgn[:, :, 2:][:, :, CALIB_GROUPS, :]

    N_calib_groups  = len(CALIB_GROUPS)
    ksp_spinav_ctgn = ksp_spinav_ctgn[:N_ch]
    ksp_stat_ngctr  = (
        (-1j) * ksp_spinav_ctgn
        .reshape((N_ch, N_t_spiral, N_calib_groups, Nt_navi, -1))
        .transpose((-2, 2, 0, 1, -1))
    )
    if select_spirals:
        ksp_stat_ngctr = ksp_stat_ngctr[..., spirals_to_select]

    ksp_stat_ngct  = ksp_stat_ngctr.reshape((Nt_navi, N_calib_groups, N_ch, -1))
    ksp_stat_ngct /= ksp_stat_ngct.size
    print('ksp_stat_ngct', ksp_stat_ngct.shape)

# %% [markdown]
# ### Phi Projection & Noise Estimation

# %%
print('img_dummy2_xyzc', img_dummy2_xyzc.shape)
Phi_navi_ct = np.squeeze(cfl.readcfl(data_path_new + dir_diff_poses[0] + 'Phi_spinav_cc')).T
if select_spirals:
    # select the same spiral positions used for k-space: (N_coef, Nt_navi, N_spir_total) → subset
    Phi_navi_ct = Phi_navi_ct.reshape((5, Nt_navi, -1))[..., spirals_to_select].reshape((5, -1))
img_dummy2_xyzt = np.matmul(img_dummy2_xyzc, Phi_navi_ct)
print('img_dummy2_xyzt', img_dummy2_xyzt.shape)

# %%
nc         = 5
ksp_tgcr   = cfl.readcfl(data_path_new + dir_diff_poses[0] + 'ksp_full_cc').squeeze()
noise_data = ksp_tgcr[-N_NOISE_FRAMES:, 0, :nc, :].transpose(1, 0, -1).reshape(nc, -1)
noise_cov  = noise_data @ np.conj(noise_data).T / noise_data.shape[-1]  # diagnostic only
power_real = np.mean(signal_power(ksp_tgcr[:, [0]].transpose(-1, 1, -2, 0)))
del ksp_tgcr

pl.ImagePlot(noise_cov, colormap='gray')

# %% [markdown]
# ### SVD Dictionary – Small Grid for Basis Estimation

# %%
_grid_keys  = ['shifts_x', 'shifts_y', 'shifts_z', 'angles_x', 'angles_y', 'angles_z']
_step_rad   = DICT_STEP_DEG * np.pi / 180
_step_range = np.arange(-DICT_MAX_STEPS, DICT_MAX_STEPS + 1, 1)

svd_gendict = {
    'angles_x': _step_range * _step_rad,
    'angles_y': _step_range * _step_rad,
    'angles_z': _step_range * _step_rad,
    'shifts_x': _step_range * DICT_STEP_PIXEL,
    'shifts_y': _step_range * DICT_STEP_PIXEL,
    'shifts_z': _step_range * DICT_STEP_PIXEL,
}
motion_mp_svd = np.stack(np.meshgrid(*[svd_gendict[k] for k in _grid_keys]), axis=-1).reshape(-1, 6)

svd_gendict['motion_mp']    = motion_mp_svd
svd_gendict['n_pars_total'] = motion_mp_svd.shape[0]
print(f'SVD grid: {motion_mp_svd.shape[0]} motion states')

with open(f'{saveDIR}input_gendict_{save_suffix}_forSVD.pkl', 'wb') as fh:
    pickle.dump(svd_gendict, fh)

# %% [markdown]
# ### SVD Basis Computation

# %%
clean_gpu(device)
N_coef = 5   # interTR: 5 subspace coefficients

if do_SVD_dictionary:
    N_mo_svd         = motion_mp_svd.shape[0] + 1
    motion_to_sim_mp = np.zeros((N_mo_svd, 6))
    motion_to_sim_mp[1:] = motion_mp_svd

    svd_gendict['dict_fullname'] = saveDIR + f'qdict_forSVD_{save_suffix}'

    if generate_calib_dictionary:
        tic = time.perf_counter()
        with device:
            clean_gpu(device)
            svd_gendict.update({
                'input_img_xyzq': mvd(img_dummy2_xyzc),
                'device':         device,
                'motion_mp':      mvd(motion_to_sim_mp),
                'sens_xyzc':      mvd(sens_dummy_cxyz).transpose((1, 2, 3, 0)),
                'k_coords_ka':    mvd(coords_navi_c_tr).astype(float).T,
                'Phi_qnr':        mvd(Phi_navi_ct).reshape((N_coef, -1, N_navi_tr)),
                'rot_center':     'center',
            })
            dict_gen(svd_gendict, debug=0, coils_before_motion=False)
        print(f'SVD dict done in {time.perf_counter()-tic:.1f}s')

    dict_ksp_mctq   = np.memmap(svd_gendict['dict_fullname'] + '.dat', dtype=np.complex64,
                                mode='r', shape=(N_mo_svd, N_ch, N_t, Nt_navi))
    dict_mo_pars_mp = np.memmap(svd_gendict['dict_fullname'] + '_moIdx.dat', dtype=np.float32,
                                mode='r', shape=(N_mo_svd, n_params))

# %%
if do_SVD_dictionary:
    dict_ksp_scaled_nmct = norm_min_max_zero(dict_ksp_mctq.transpose((3, 0, 1, 2)), 0)
    ksp_stat_scaled_ngct = norm_min_max_zero(ksp_stat_ngct, 0)

    print('simulated signals', dict_ksp_scaled_nmct.shape)
    print('in-vivo signals  ', ksp_stat_scaled_ngct.shape)

# %%
plot_3dimg(np.angle(img_dummy2_xyzt[..., -10]), 56)
plot_3dimg(np.angle(img_navi_end_xyzp[..., 0]), 56)

# %%
if do_SVD_dictionary and svd_compr_method in ['gEVD', 'pinv_SVD']:
    for nii in [4]:
        for ch in range(N_ch):
            plot_2signals(
                (1j) * ksp_stat_scaled_ngct[nii][0, ch, 1::N_navi_tr],
                dict_ksp_scaled_nmct[nii][0, ch, 1::N_navi_tr],
                labels=['In vivo', 'Simulated'], xlim=[0, 500],
            )

# %%
if do_SVD_dictionary and svd_compr_method in ['gEVD', 'pinv_SVD'] and calibrate_wrt_static:
    dot_sim_ncs = np.zeros((Nt_navi, N_ch, 3))
    for nii in range(Nt_navi):
        for ch in range(N_ch):
            for spir in range(3):
                dot_sim_ncs[nii, ch, spir] = dot_sim(
                    ksp_stat_ngct[nii, 0, ch, spir::N_navi_tr],
                    dict_ksp_scaled_nmct[nii, 0, ch, spir::N_navi_tr],
                )

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im = axs[0].imshow(dot_sim_ncs[:, 0, :], cmap='rainbow')
    axs[0].set_title('Sim. similarity: all contrasts × spirals', fontsize=9)
    fig.colorbar(im, ax=axs[0])
    im = axs[1].imshow(dot_sim_ncs[:, :, 0], cmap='rainbow')
    axs[1].set_title('Sim. similarity: all contrasts × coils', fontsize=9)
    fig.colorbar(im, ax=axs[1])

# %%
if do_SVD_dictionary and svd_compr_method == 'whiten_SVD':
    for nii in navis_to_use:
        pc = signal_power(ksp_stat_scaled_ngct[[nii]][:, [0]])
        ksp_stat_scaled_ngct[[nii]] = norm_signals_to_real(ksp_stat_scaled_ngct[[nii]], power_real, pc)
        pc = signal_power(dict_ksp_scaled_nmct[[nii]][:, [0]])
        dict_ksp_scaled_nmct[[nii]] = norm_signals_to_real(dict_ksp_scaled_nmct[[nii]], power_real, pc)

# %%
if do_SVD_dictionary and generate_calib_dictionary:
    dict_ksp_scaled_nmct = norm_min_max_zero(dict_ksp_mctq.transpose((3, 0, 1, 2)), 0)

    N_t_svd = Nt_up * N_navi_tr
    u_rtt   = np.zeros((Nt_navi, N_ch, N_t_svd, N_t_svd), dtype=np.complex64)
    s_rt    = np.zeros((Nt_navi, N_ch, N_t_svd), dtype=np.complex64)

    for nii in navis_to_use:
        clean_gpu(device)
        with device:
            dict_ksp_gpu = mvd(dict_ksp_scaled_nmct[[nii]]).transpose((1, 2, 0, 3))

            if svd_compr_method in ['gEVD', 'pinv_SVD']:
                dict_dB0_npct = mvd(ksp_stat_scaled_ngct[[nii]]).transpose((1, 2, 0, 3))
                dict_dB0_npct = dict_dB0_npct[CALIB_GROUPS, ...]
                dict_dB0_npct = xp.concatenate((dict_ksp_gpu[[0]], dict_dB0_npct), axis=0)
                print('B size', dict_dB0_npct.shape)
            else:
                dict_dB0_npct = None

            print('A size', dict_ksp_gpu.shape)
            result_svd, _ = gEVD_dictionary(dict_ksp_gpu, dict_dB0_npct, xp=xp,
                                            method=svd_compr_method, demean_data=True,
                                            unit_variance=False, scale_variance=True)

            if svd_compr_method in ['SVD', 'pinv_SVD']:
                s_rt[nii]  = np.tile(result_svd[1].get(), (N_ch, 1))
                u_rtt[nii] = np.tile(result_svd[0].get(), (N_ch, 1, 1))
            else:
                s_rt[nii]  = result_svd[0]
                u_rtt[nii] = result_svd[1]

    cfl.writecfl(f'{saveDIR}dictgator/u_nctt_{save_suffix}', u_rtt)
    cfl.writecfl(f'{saveDIR}dictgator/s_nct_{save_suffix}',  s_rt)

# %%
if do_SVD_dictionary:
    u_nrtt = cfl.readcfl(f'{saveDIR}dictgator/u_nctt_{save_suffix}')
    print('u_nrtt', u_nrtt.shape)

# %% [markdown]
# ### Full Dictionary Grid

# %%
full_gendict = {
    'angles_x': np.arange(np.min(mo_GT[:, 3]), np.max(mo_GT[:, 3]) + _step_rad,        _step_rad),
    'angles_y': np.arange(np.min(mo_GT[:, 4]), np.max(mo_GT[:, 4]) + _step_rad,        _step_rad),
    'angles_z': np.arange(np.min(mo_GT[:, 5]), np.max(mo_GT[:, 5]) + _step_rad,        _step_rad),
    'shifts_x': np.arange(np.min(mo_GT[:, 0]), np.max(mo_GT[:, 0]) + DICT_STEP_PIXEL,  DICT_STEP_PIXEL),
    'shifts_y': np.arange(np.min(mo_GT[:, 1]), np.max(mo_GT[:, 1]) + DICT_STEP_PIXEL,  DICT_STEP_PIXEL),
    'shifts_z': np.arange(np.min(mo_GT[:, 2]), np.max(mo_GT[:, 2]) + DICT_STEP_PIXEL,  DICT_STEP_PIXEL),
}
motion_mp = np.stack(np.meshgrid(*[full_gendict[k] for k in _grid_keys]), axis=-1).reshape(-1, 6)
motion_mp = np.concatenate((motion_mp, mo_GT), axis=0)

full_gendict['motion_mp']    = motion_mp
full_gendict['n_pars_total'] = motion_mp.shape[0]
print(f'Full grid: {motion_mp.shape[0]} motion states')

# %% [markdown]
# ### Full Dictionary Generation

# %%
if generate_dictionary:
    with open(f'{saveDIR}input_gendict_{save_suffix}.pkl', 'wb') as fh:
        pickle.dump(full_gendict, fh)
else:
    with open(f'{saveDIR}input_gendict_{save_suffix}.pkl', 'rb') as fh:
        full_gendict = pickle.load(fh)
print('motion_mp', full_gendict['motion_mp'].shape)

if generate_dictionary:
    for nii in navis_to_use:
        with device:
            full_gendict.update({
                'dict_fullname':  saveDIR + f'qdict_full_nii{nii}_{save_suffix}',
                'input_img_xyzq': mvd(img_dummy2_xyzt[..., [3 * nii]]),
                'device':         device,
                'sens_xyzc':      sens_dummy_cxyz.transpose((1, 2, 3, 0)),
                'k_coords_ka':    coords_navi_c_tr[:, N_navi_tr * Nt_lo : N_navi_tr * Nt_up].astype(float).T,
                'rot_center':     'center',
            })
            if do_SVD_dictionary:
                full_gendict['v_ct_tnew'] = mvd(u_nrtt[nii])[..., Nt_new_lo:Nt_new_up]

            clean_gpu(device)
            tic = time.perf_counter()
            dict_gen(full_gendict, debug=0, coils_before_motion=False)
            print(f'nii={nii} done in {time.perf_counter()-tic:.1f}s')

# %% [markdown]
# ### Signal Power Normalisation (whiten_SVD only)

# %%
if svd_compr_method == 'whiten_SVD':
    for nii in navis_to_use:
        power_current = signal_power(ksp_navi_ngc_tr[[nii]][:, [0]])
        ksp_navi_ngc_tr[[nii]] = norm_signals_to_real(ksp_navi_ngc_tr[[nii]], power_real, power_current)
    # power_current (last navigator's power) stays in scope for run_matching's dict normalisation

# %% [markdown]
# ### Dictionary Matching

# %%
res_match_dict, motion_Est, ksp_compr_gct, dict_compr_mct, dict_mo_par_mp = run_matching(
    ksp_navi_ngc_tr, full_gendict, mo_GT, img_navi_end_xyzp,
    navis_to_use   = navis_to_use,
    Nt_new_lo      = Nt_new_lo,
    Nt_new_up      = Nt_new_up,
    saveDIR        = saveDIR,
    save_suffix    = save_suffix,
    device         = device,
    do_SVD_dictionary = do_SVD_dictionary,
    svd_compr_method  = svd_compr_method,
    u_nrtt         = u_nrtt if do_SVD_dictionary else None,
    lamda_kalman   = 1e-2,
    n_minima       = 30,
    power_real     = power_real,
    power_current  = power_current if svd_compr_method == 'whiten_SVD' else None,
)

motion_Est[0] = np.copy(motion_Est[2])  # navi 0 has low signal; copy nearest reliable estimate

motion_Est_gnp = motion_Est.transpose((1, 0, 2))
motion_GT_gnp  = np.tile(mo_GT[:, None, :], (1, Nt_navi, 1))
print(motion_Est_gnp.shape, motion_GT_gnp.shape)

res_match = {
    'motion_Est': motion_Est_gnp.reshape((-1, 6)),
    'motion_GT':  motion_GT_gnp.reshape((-1, 6)),
}

dict_name_to_save = dir_matching + f'resMatch_{save_suffix}_{figure_save_suffix}.pkl'
with open(dict_name_to_save, 'wb') as fh:
    pickle.dump(res_match, fh)
print(f'Results saved → {dict_name_to_save}')

# %% [markdown]
# ### Metrics & Visualisation

# %%
motion_dict_last = res_match['motion_Est'][Nt_navi - 1::Nt_navi].copy()
motion_afni_last = res_match['motion_GT'][Nt_navi - 1::Nt_navi].copy()

for arr in [motion_afni_last, motion_dict_last]:
    arr[:, :3] *= 4.
    arr[:, 3:] *= 180. / np.pi

metrics = compute_motion_metrics(motion_afni_last, motion_dict_last)
print(metrics)

# %%
res_match['motion_Est'].shape, res_match['motion_GT'].shape, N_GR_scan, Nt_navi

# %%
ser_ii = 0
groups_to_match = slice(ser_ii * N_GR_scan, (ser_ii + 1) * N_GR_scan)
res_match_ser   = {
    'motion_Est': motion_Est_gnp[groups_to_match].reshape((-1, 6)),
    'motion_GT':  motion_GT_gnp[groups_to_match].reshape((-1, 6)),
}

figname_traj = f'{dir_figure}/resMatch_motion_{save_suffix}_{figure_save_suffix}'
plot_motion_traj(
    res_match_ser, img_resolution=4, N_GR=N_GR_scan, Nt_navi=Nt_navi,
    save=False, figname=figname_traj, plot_smooth_noGT=False,
    figsize=(10, 5), translation_lims=[-2, 2], rotation_lims=[-4, 4],
    markers=['-o', '-o'], labels_methods=['DM', 'GT'],
)

# %% [markdown]
# ### (Optional) Export for External Pipelines

# %%
save_data = {
    'ksp_gnct':       ksp_navi_ngc_tr.transpose((1, 0, 2, 3)),
    'coords_at':      coords_navi_c_tr,
    'motion_Est_gnp': motion_Est_gnp,
    'sens_cxyz':      sens_dummy_cxyz,
    'img_nxyz':       img_dummy2_xyzt[..., 1::3].transpose((-1, 0, 1, 2)),
    'mo_GT_gp':       mo_GT,
    'u_H_nctt':       u_nrtt[..., Nt_new_lo:Nt_new_up],
}
_mat_path = f'{SAVEMAT_DIR}data_{save_suffix}_{figure_save_suffix}.mat'
savemat(_mat_path.removesuffix('.mat'), save_data)   # scipy adds .mat automatically
print(f'Data exported → {_mat_path}')
print()
print('To refine with gradient optimisation (per series):')
for _ser in range(len(dir_diff_poses)):
    print(f'  python Motion_optim_GT.py \\')
    print(f'    --input {_mat_path} \\')
    print(f'    --date {date_tag} --series-ii {_ser} \\')
    print(f'    --save-dir-match {dir_matching} \\')
    print(f'    --save-dir-fig {dir_figure}')
