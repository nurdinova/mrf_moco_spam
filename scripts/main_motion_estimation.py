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
# | 1. Data loading from cfl-files | Navigator k-space, coil sensitivities, trajectory and image navigators across series. |
# | 2. Dummy-navigator calibration | Navigators of one dummy group (Q-scout, no motion) are the measured no-motion signal; used to compare simulated and in-vivo signals and as the whiten-SVD reference. |
# | 3. Subspace projection | Q-scout coefficients are projected into the MRF temporal subspace → per-contrast input for motion-dictionary simulation. |
# | 4. Motion SVD basis | Small motion dictionary (±1 vox / ±0.5°) decomposed into a per-channel time-compression basis **U**. |
# | 5. Dictionary generation | Fine 6-DOF grid simulated per navigator contrast, stored as memory-mapped files compressed through **U**. |
# | 6. Dictionary matching | Measured navigators projected through **U**, matched via complex dot-product. Optional regularisation towards the previous navigator. |
# | 7. Evaluation | Estimated trajectories vs. image nav registration-based ground-truth; correlation and MAD reported. Results saved for further motion optimization and reconstruction. |
#

# %%
import argparse
import json
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import seaborn as sns
import sigpy as sp
import sigpy.plot as pl

sys.path.append("/usr/local/app/bart/bart-0.6.00/python/")
sys.path.append("/local_mount/space/mayday/data/users/aizadan/dictgator/")
sys.path.append("/local_mount/space/mayday/data/users/aizadan/mrf_moco_spam/src/")

import cfl
from motion_estimation import time_compression_basis
from src.analysis.metrics import compute_motion_metrics
from src.analysis.plot_dictMatch import dot_sim, plot_2signals, plot_3dimg
from src.analysis.plot_motion import plot_motion_traj
from src.dict_gen_gpu import dict_gen
from src.dict_match import match_navi
from src.Forward_NavAcq_2 import call_afni
from src.simulate_Nav_signal import load_data
from src.utils import apply_svd_basis, clean_gpu, norm_min_max_zero

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
# ### Helper Functions

# %%
_DESCRIPTION = (
    "Motion estimation from spiral navigators (3 TRs, ~12 ms each) interleaved\n"
    "every ~40 TR within a 3D SPI MRF sequence.\n\n"
    "Pipeline: data loading → dummy-navigator calibration → subspace projection\n"
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
    p.add_argument('--series',             type=int, nargs='+', default=[0, 1],
                   help='Indices into the dataset\'s Series list; the first one holds the scout and dummy navigators')
    p.add_argument('--generate-dict',      action='store_true', help='Generate full motion dictionary')
    p.add_argument('--generate-calib-dict',action='store_true', help='Generate calibration dictionary for SVD basis')
    p.add_argument('--no-svd',             action='store_false', dest='do_svd',
                   help='Skip SVD time-compression of navigator signals')
    p.add_argument('--svd-method',         default='whiten_SVD', choices=['whiten_SVD', 'SVD'])
    p.add_argument('--gpu',                type=int, default=2, help='CUDA device index')
    p.add_argument('--save-suffix',        default=None, help='Override auto-generated save suffix')
    p.add_argument('--results-name',       default=None,
                   help='File name of the saved trajectories in dir_matching (default: resMatch_<suffix>_<figure suffix>.pkl)')
    p.add_argument('--savemat-dir',        default='./', help='Directory of the exported .mat')
    p.add_argument('--n-navi-tr',          type=int, default=None,
                   help='Spirals per navigator, overrides N_navi_tr from data_info.json')
    p.add_argument('--registration-per-series', action='store_true',
                   help='Registration of each series relative to its own first group (single-series runs, '
                        'where scout and calibration come from that series) instead of the first series')
    p.set_defaults(do_svd=True)
    return p

def _in_notebook():
    # 'ipykernel' in sys.modules is not enough: sigpy.plot imports ipykernel
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        return False

def mvc(x):
    return sp.to_device(x, sp.cpu_device)

def mvd(x):
    # `device` is set in the Device & Paths cell
    return sp.to_device(x, device)

def load_registration(series_index, data_dir, n_groups):
    # motion_GT_all.npy holds n_groups groups per series, in the order of metadata['Series'];
    # otherwise one file per series
    path_all = f'{data_dir}dictgator/motion_GT_all.npy'
    if os.path.exists(path_all):
        return np.load(path_all)[series_index * n_groups : (series_index + 1) * n_groups]
    return np.load(f'{data_dir}dictgator/motion_GT_ser{series_index}.npy')[:n_groups]

def make_grid(values, step):
    low, high = np.min(values), np.max(values)
    if low == high:   # no motion along this axis -> ±1 step around the value
        return np.array([low - step, low, low + step])
    return np.arange(low, high + step, step)

# %% [markdown]
# ### Configuration
#
# Run as a notebook: `args` defaults are used.
# Run as a script:   `python main_motion_estimation.py --date 250425 --series 0 1 --gpu 2 ...`

# %%
args = _build_parser().parse_args([] if _in_notebook() else None)

# %%
# ---- parameters (edit for interactive runs) ----
date_tag        = args.date
dataset_number  = args.dataset
series_indices  = args.series

generate_dictionary      = args.generate_dict
generate_calib_dictionary = args.generate_calib_dict
do_SVD_dictionary        = args.do_svd
svd_compr_method         = args.svd_method

# flags not exposed to CLI
select_spirals       = True
coils_to_select      = [0, 1, 2, 3, 4]
navis_to_use         = None        # navigator contrasts to match; None = all
n_params             = 6           # motion DOF
do_register_img_nav  = False
lamda_reg            = 0.          # regularisation weight towards the previously matched navigator; 0 = off
n_minima             = 30
figure_save_suffix   = 'noKalman_smallDict'

# k-space / readout geometry
N_READOUT_PER_SPIRAL = 1658   # readout points per spiral arm
N_NOISE_FRAMES       = 200    # trailing frames used for noise covariance

# dummy groups (Q-scout, no motion) acquired before the MRF groups
N_DUMMY_GROUPS       = None   # None = from the data: navigator groups - image-navigator groups
CALIB_DUMMY_GROUP    = 1      # its navigators are the measured no-motion signal (calibration, whiten_SVD)
                              # without that dummy group: the first group of the first (static) series

# navigator reliability: navigators whose signal is not consistent across groups are not matched;
# their trajectories are interpolated from the neighbouring navigators
NAVI_CORRELATION_THRESHOLD = 0.9   # mean |correlation| of the navigator signal across groups
MAX_EXCLUDED_NAVIS         = 2

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
# regularisation distance in grid steps, so that shifts [vox] and rotations [rad] are penalised alike
REG_SCALE_P     = np.array(3 * [DICT_STEP_PIXEL] + 3 * [DICT_STEP_DEG * np.pi / 180])

# output path for external pipeline export
SAVEMAT_DIR = args.savemat_dir

# %% [markdown]
# ### Device & Paths

# %%
device = sp.Device(args.gpu)
xp     = device.xp

# %%
DICTGATOR_DIR = '/local_mount/space/mayday/data/users/aizadan/dictgator/'
JSON_FILENAME = os.path.join(DICTGATOR_DIR, 'data_info.json')

with open(JSON_FILENAME, 'r') as fh:
    metadata = json.load(fh)[date_tag][dataset_number]

dir_matching  = metadata['dir_matching']
dir_figure    = metadata['dir_figures']
saveDIR       = metadata['saveDIR']
data_path_new = metadata['recon_data_path']
N_navi_tr     = args.n_navi_tr or metadata['N_navi_tr']

dir_diff_poses = [f'Series{s}/' for s in metadata['Series']]
dir_diff_poses = [dir_diff_poses[s] for s in series_indices]

save_suffix = args.save_suffix or f'final_{svd_compr_method}_DictMatch_dummy_calib'

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
        img_dummy2_xyzc, sens_dummy_xyzc, coords_navi_c_tr, _, img_navi_end_xyzp = data
        if N_DUMMY_GROUPS is None:
            # dummy groups have navigators but no image navigator
            N_DUMMY_GROUPS = data[3].shape[1] - img_navi_end_xyzp.shape[-1]
        print('dummy groups with navigators:', N_DUMMY_GROUPS)
    ksp_navi = data[3][:, N_DUMMY_GROUPS:, :5, ...]
    ksp_navi_ngc_tr = ksp_navi if ser_ii == 0 \
                      else np.concatenate((ksp_navi_ngc_tr, ksp_navi), axis=1)

print('ksp_navi_ngc_tr', ksp_navi_ngc_tr.shape)

ksp_navi_ngc_tr = ksp_navi_ngc_tr[:, :, coils_to_select]
sens_dummy_xyzc = sens_dummy_xyzc[..., coils_to_select]

Nt_navi, N_GR, N_ch, N_t = ksp_navi_ngc_tr.shape
nx, ny, nz = img_navi_end_xyzp.shape[:3]
N_GR_scan  = N_GR // len(dir_diff_poses)

if navis_to_use is None:
    navis_to_use = list(range(Nt_navi))

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
print('ksp_navi_ngct    ', ksp_navi_ngc_tr.shape)

# %% [markdown]
# ### Navigator Reliability
#
# Per navigator index: mean |correlation| of the navigator signal between all pairs of groups of the
# first (static) series, so that real motion in the other series does not count as unreliability.
# Navigators below `NAVI_CORRELATION_THRESHOLD` (at most `MAX_EXCLUDED_NAVIS`, lowest first) are not
# matched; their trajectories are interpolated from the neighbouring navigators of the same group.

# %%
navi_correlation_n = np.zeros(Nt_navi)
off_diagonal = ~np.eye(N_GR_scan, dtype=bool)
for n_ii in range(Nt_navi):
    signals_g_ct = ksp_navi_ngc_tr[n_ii, :N_GR_scan].reshape((N_GR_scan, -1))
    signals_g_ct = signals_g_ct / np.linalg.norm(signals_g_ct, axis=1, keepdims=True)
    navi_correlation_n[n_ii] = np.abs(signals_g_ct.conj() @ signals_g_ct.T)[off_diagonal].mean()

navis_unreliable = [int(n_ii) for n_ii in np.argsort(navi_correlation_n)
                    if navi_correlation_n[n_ii] < NAVI_CORRELATION_THRESHOLD][:MAX_EXCLUDED_NAVIS]
navis_to_use = [n_ii for n_ii in navis_to_use if n_ii not in navis_unreliable]

print('navigator correlation across groups:', np.array2string(navi_correlation_n, precision=3, max_line_width=300))
print('excluded navigators:', navis_unreliable, '→ matched:', navis_to_use)

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
mo_GT_per_series = [load_registration(s, data_path_new, N_GR_scan) for s in series_indices]
if args.registration_per_series:
    mo_GT_per_series = [mo_GT_s - mo_GT_s[0] for mo_GT_s in mo_GT_per_series]
mo_GT = np.concatenate(mo_GT_per_series, axis=0)
print('mo_GT', mo_GT.shape)

# %% [markdown]
# ### Dummy-Group Navigators (Calibration)
#
# Navigators acquired during the dummy groups of the first series (Q-scout, no motion), in raw k-space units.
# Group `CALIB_DUMMY_GROUP` is the measured no-motion signal. Datasets without that dummy group use the
# first group of the first (static) series instead.

# %%
ksp_spinav_ctgn = cfl.readcfl(data_path_new + dir_diff_poses[0] + 'ksp_spinav_ctgr_cc')[coils_to_select]

if N_DUMMY_GROUPS > CALIB_DUMMY_GROUP:
    calibration_group  = CALIB_DUMMY_GROUP
    calibration_source = f'dummy group {CALIB_DUMMY_GROUP}'
else:
    calibration_group  = N_DUMMY_GROUPS
    calibration_source = 'first group of the static series (no dummy navigators)'
n_raw_groups = max(N_DUMMY_GROUPS, calibration_group + 1)
print('calibration signal:', calibration_source)

ksp_spinav_ctgn = ksp_spinav_ctgn[:N_ch, :N_READOUT_PER_SPIRAL, :n_raw_groups]
ksp_raw_gnctr = (
    (-1j) * ksp_spinav_ctgn
    .reshape((N_ch, N_READOUT_PER_SPIRAL, n_raw_groups, Nt_navi, -1))
    .transpose((2, 3, 0, 1, 4))
)
if select_spirals:
    ksp_raw_gnctr = ksp_raw_gnctr[..., spirals_to_select]
ksp_raw_gnct   = ksp_raw_gnctr.reshape((n_raw_groups, Nt_navi, N_ch, -1))
ksp_dummy_gnct = ksp_raw_gnct[:N_DUMMY_GROUPS]
ksp_calib_nct  = ksp_raw_gnct[calibration_group]
del ksp_spinav_ctgn, ksp_raw_gnctr, ksp_raw_gnct

print('ksp_dummy_gnct', ksp_dummy_gnct.shape)
print('ksp_calib_nct ', ksp_calib_nct.shape)

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
nc         = len(coils_to_select)
ksp_tgcr   = cfl.readcfl(data_path_new + dir_diff_poses[0] + 'ksp_full_cc').squeeze()
noise_data = ksp_tgcr[-N_NOISE_FRAMES:, 0][:, coils_to_select].transpose(1, 0, -1).reshape(nc, -1)
noise_cov  = noise_data @ np.conj(noise_data).T / noise_data.shape[-1]  # per sample, raw k-space units (whiten_SVD)
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
    dict_ksp_scaled_nmct  = norm_min_max_zero(dict_ksp_mctq.transpose((3, 0, 1, 2)), 0)
    ksp_calib_scaled_nmct = norm_min_max_zero(ksp_calib_nct[:, None], 0)
    print('simulated signals', dict_ksp_scaled_nmct.shape)
    print('in-vivo signals  ', ksp_calib_scaled_nmct.shape)

# %%
plot_3dimg(np.angle(img_dummy2_xyzt[..., -10]), 56)
plot_3dimg(np.angle(img_navi_end_xyzp[..., 0]), 56)

# %%
if do_SVD_dictionary:
    for nii in [4]:
        for ch in range(N_ch):
            plot_2signals(
                (1j) * ksp_calib_scaled_nmct[nii][0, ch, 1::N_navi_tr],
                dict_ksp_scaled_nmct[nii][0, ch, 1::N_navi_tr],
                labels=['In vivo', 'Simulated'], xlim=[0, 500],
            )

# %%
if do_SVD_dictionary:
    dot_sim_ncs = np.zeros((Nt_navi, N_ch, 3))
    for nii in range(Nt_navi):
        for ch in range(N_ch):
            for spir in range(3):
                dot_sim_ncs[nii, ch, spir] = dot_sim(
                    ksp_calib_nct[nii, ch, spir::N_navi_tr],
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
# only the first Nt_new_up basis vectors are kept: every later step uses u[..., Nt_new_lo:Nt_new_up]
if do_SVD_dictionary and generate_calib_dictionary:
    u_nctk = np.zeros((Nt_navi, N_ch, N_t, Nt_new_up), dtype=np.complex64)
    s_nct  = np.zeros((Nt_navi, N_ch, N_t), dtype=np.complex64)

    for nii in navis_to_use:
        clean_gpu(device)
        with device:
            dict_ksp_gpu = mvd(dict_ksp_scaled_nmct[[nii]]).transpose((1, 2, 0, 3))

            if svd_compr_method == 'whiten_SVD':
                # raw k-space units so that noise_cov applies as-is
                calib_signal_ct = mvd(ksp_calib_nct[nii])
                # complex least-squares fit of the simulated no-motion signal to the measured one:
                # matches scale and global phase, otherwise the mismatch is dominated by the phase offset
                simulated_static_ct = dict_ksp_gpu[0, :, 0]
                dict_ksp_gpu = dict_ksp_gpu * (xp.vdot(simulated_static_ct, calib_signal_ct)
                                               / xp.vdot(simulated_static_ct, simulated_static_ct))
                # rescale to unit RMS to stay within complex64 precision
                scale = float(xp.linalg.norm(calib_signal_ct)) / np.sqrt(calib_signal_ct.size)
                dict_ksp_gpu        = dict_ksp_gpu / scale
                calib_signal_ct     = calib_signal_ct / scale
                noise_covariance_cc = mvd(noise_cov) / scale**2
            else:
                calib_signal_ct     = None
                noise_covariance_cc = None

            print(f'nii={nii} dictionary', dict_ksp_gpu.shape)
            basis_tt, values = time_compression_basis(dict_ksp_gpu, calib_signal_ct, noise_covariance_cc,
                                                      method=svd_compr_method, xp=xp)

            s_nct[nii]  = np.tile(mvc(values), (N_ch, 1))
            u_nctk[nii] = np.tile(mvc(basis_tt[:, :Nt_new_up]), (N_ch, 1, 1))

    cfl.writecfl(f'{saveDIR}dictgator/u_nctk_{save_suffix}', u_nctk)
    cfl.writecfl(f'{saveDIR}dictgator/s_nct_{save_suffix}',  s_nct)

# %%
if do_SVD_dictionary:
    u_nctk = cfl.readcfl(f'{saveDIR}dictgator/u_nctk_{save_suffix}')
    s_nct  = cfl.readcfl(f'{saveDIR}dictgator/s_nct_{save_suffix}')
    print('u_nctk', u_nctk.shape)

# %% [markdown]
# ### Full Dictionary Grid

# %%
full_gendict = {
    'angles_x': make_grid(mo_GT[:, 3], _step_rad),
    'angles_y': make_grid(mo_GT[:, 4], _step_rad),
    'angles_z': make_grid(mo_GT[:, 5], _step_rad),
    'shifts_x': make_grid(mo_GT[:, 0], DICT_STEP_PIXEL),
    'shifts_y': make_grid(mo_GT[:, 1], DICT_STEP_PIXEL),
    'shifts_z': make_grid(mo_GT[:, 2], DICT_STEP_PIXEL),
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
                'input_img_xyzq': mvd(img_dummy2_xyzt[..., [N_navi_tr * nii]]),
                'device':         device,
                'sens_xyzc':      sens_dummy_cxyz.transpose((1, 2, 3, 0)),
                'k_coords_ka':    coords_navi_c_tr[:, N_navi_tr * Nt_lo : N_navi_tr * Nt_up].astype(float).T,
                'rot_center':     'center',
            })
            if do_SVD_dictionary:
                full_gendict['v_ct_tnew'] = mvd(u_nctk[nii])[..., Nt_new_lo:Nt_new_up]

            clean_gpu(device)
            tic = time.perf_counter()
            dict_gen(full_gendict, debug=0, coils_before_motion=False)
            print(f'nii={nii} done in {time.perf_counter()-tic:.1f}s')

# %% [markdown]
# ### Dictionary Matching

# %%
if not do_SVD_dictionary:
    # uncompressed dictionary: match on the full simulated time window
    Nt_new_lo = N_navi_tr * Nt_lo
    Nt_new_up = N_navi_tr * Nt_up
Nt_compr = Nt_new_up - Nt_new_lo

motion_Est = np.zeros((Nt_navi, N_GR, n_params))
for nii in navis_to_use:
    full_gendict['dict_fullname'] = saveDIR + f'qdict_full_nii{nii}_{save_suffix}'
    n_pars = full_gendict['n_pars_total']
    dict_ksp_mct1   = np.memmap(full_gendict['dict_fullname'] + '.dat', dtype=np.complex64,
                                mode='r', shape=(n_pars, N_ch, Nt_compr, 1))
    dict_mo_pars_mp = np.memmap(full_gendict['dict_fullname'] + '_moIdx.dat', dtype=np.float32,
                                mode='r', shape=(n_pars, n_params))
    # regularisation prior: the previously matched navigator (unmatched ones are still zero at this point)
    navis_matched_before = [n_ii for n_ii in navis_to_use if n_ii < nii]
    mo_prev_gp = motion_Est[navis_matched_before[-1]] if (lamda_reg and navis_matched_before) else None

    with device:
        ksp_navi_gpu = mvd(ksp_navi_ngc_tr[[nii], ..., N_navi_tr * Nt_lo : N_navi_tr * Nt_up])
        if do_SVD_dictionary:
            # project before matching so that match_navi works on the compressed time axis
            ksp_navi_gpu = apply_svd_basis(ksp_navi_gpu, mvd(u_nctk[nii][..., Nt_new_lo:Nt_new_up]))
        res_match_dict = match_navi(ksp_navi_gpu, dict_ksp_mct1, dict_mo_pars_mp, None, full_gendict,
                                    mo_GT=mo_GT, img_navi_xyzp=img_navi_end_xyzp, match_mode='DictMatch',
                                    lamda_reg=lamda_reg, reg_scale_p=REG_SCALE_P, mo_prev_gp=mo_prev_gp,
                                    n_minima=n_minima)
    motion_Est[nii] = np.copy(res_match_dict['motion_Est'])
    print(f'nii={nii} matched')

# navigators that were not matched: interpolate from the neighbouring navigators of the same group
navis_not_matched = [n_ii for n_ii in range(Nt_navi) if n_ii not in navis_to_use]
if navis_not_matched:
    for g_ii in range(N_GR):
        for p_ii in range(n_params):
            motion_Est[:, g_ii, p_ii] = np.interp(np.arange(Nt_navi), navis_to_use, motion_Est[navis_to_use, g_ii, p_ii])
    print('interpolated navigators:', navis_not_matched)

motion_Est_gnp = motion_Est.transpose((1, 0, 2))
motion_GT_gnp  = np.tile(mo_GT[:, None, :], (1, Nt_navi, 1))
print(motion_Est_gnp.shape, motion_GT_gnp.shape)

res_match = {
    'motion_Est':         motion_Est_gnp.reshape((-1, 6)),
    'motion_GT':          motion_GT_gnp.reshape((-1, 6)),
    'navis_used':         np.array(navis_to_use),
    'navi_correlation_n': navi_correlation_n,
    'calibration_source': calibration_source,
}

dict_name_to_save = dir_matching + (args.results_name or f'resMatch_{save_suffix}_{figure_save_suffix}.pkl')
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
    'ksp_dummy_gnct': ksp_dummy_gnct,    # dummy-group navigators, same spirals/layout/phase as ksp_gnct
    'coords_at':      coords_navi_c_tr,
    'motion_Est_gnp': motion_Est_gnp,
    'sens_cxyz':      sens_dummy_cxyz,
    'img_nxyz':       img_dummy2_xyzt[..., N_navi_tr * np.arange(Nt_navi)].transpose((-1, 0, 1, 2)),  # same images as dict_gen
    'mo_GT_gp':       mo_GT,
    'noise_cov':      noise_cov,         # (C, C), raw k-space units -> SNR weighting in Motion_optim_GT.py
    'navis_used':     np.array(navis_to_use),   # reliable navigators; motion_optim.py optimises only these
}
if do_SVD_dictionary:
    save_data['u_H_nctt'] = u_nctk[..., Nt_new_lo:Nt_new_up]
    if svd_compr_method == 'whiten_SVD':
        save_data['lda_eigvals_nk'] = s_nct[:, 0, Nt_new_lo:Nt_new_up].real   # generalized eigenvalues per contrast

_mat_path = f'{SAVEMAT_DIR}data_{save_suffix}_{figure_save_suffix}.mat'
savemat(_mat_path, save_data)
print(f'Data exported → {_mat_path}')
print()
print('To refine with gradient optimisation (per series):')
for _ser in range(len(dir_diff_poses)):
    print(f'  python Motion_optim_GT.py \\')
    print(f'    --input {_mat_path} \\')
    print(f'    --date {date_tag} --series-ii {_ser} \\')
    print(f'    --save-dir-match {dir_matching} \\')
    print(f'    --save-dir-fig {dir_figure}')
