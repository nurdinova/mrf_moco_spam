#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ### Autodiff-based motion refinement of dictionary-matching estimates
#
# Loads the `.mat` exported by `main_motion_estimation.py`, runs a PyTorch forward-model
# optimisation (k-space data consistency + L2 similarity to the reference motion), and saves
# refined motion trajectories.
#
# The reference motion is the image-navigator registration, sampled at the end of each group.
# It is linearly interpolated to every navigator and only its shape is enforced: the constant
# offset between the estimates and the reference is removed before the L2.
#
# Navigators excluded with `--navis-to-use` (e.g. a low-SNR one) are not optimised; their motion is
# linearly interpolated from the neighbouring navigators of the same group in the saved trajectory.
#
# | `--init` | Description |
# |----------|-------------|
# | `zero` (default) | initialise from zero motion |
# | `est` | initialise from the dictionary-matching estimate |
#
# Run as a notebook: edit the parameter cell below.
# Run as a script:   `python motion_optim.py --input data.mat --date 250425 --series-ii 0 ...`

# %% [markdown]
# ### Imports

# %%
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import sigpy as sp
import torch
import torch.fft
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append("/usr/local/app/bart/bart-0.6.00/python/")
sys.path.append("/local_mount/space/mayday/data/users/aizadan/dictgator/")
from src.analysis.plot_motion import plot_motion_traj
from src.torch_optim import Forward_Motion, complex_dot_loss

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
# Run as a script:   `python motion_optim.py --input <path> --date 250425 ...`

# %%
_DESCRIPTION = (
    "Gradient-based motion refinement of dictionary-matching estimates.\n\n"
    "Loads a .mat exported by main_motion_estimation.py, runs a PyTorch forward-model\n"
    "optimisation (data consistency + L2 similarity to the interpolated reference\n"
    "motion), and saves refined motion trajectories."
)


def _build_parser():
    p = argparse.ArgumentParser(
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- input ---
    p.add_argument('--input',     default='./data_final_whiten_SVD_DictMatch_dummy_calib_noKalman_smallDict.mat',
                   type=str, help='Path to .mat exported by main_motion_estimation.py')
    p.add_argument('--date',      default='250425', type=str,
                   help='Scan date tag (output directories from data_info.json)')
    p.add_argument('--dataset',   default=0, type=int, help='Dataset index in data_info.json')
    p.add_argument('--series-ii', default=0, type=int,
                   help='Series index within the .mat: selects groups [N_GR_scan*i : N_GR_scan*(i+1)]')

    # --- compute ---
    p.add_argument('--gpu', default=2, type=int, help='CUDA device index')

    # --- initialisation ---
    p.add_argument('--init', default='zero', choices=['zero', 'est'],
                   help="Initial motion: 'zero' = all zeros, 'est' = dictionary-matching estimate")

    # --- data ---
    p.add_argument('--no-svd', action='store_false', dest='do_svd',
                   help='Skip SVD time-compression of navigator signals')
    p.set_defaults(do_svd=True)
    p.add_argument('--navis-to-use', default=None, type=str,
                   help="Navigator indices to optimise, e.g. '0-12' or '0,2-12'; the others are linearly "
                        "interpolated in the saved trajectory (default: the reliable navigators stored in the .mat)")
    p.add_argument('--N-GR-scan', default=24, type=int, help='Groups per acquisition series')
    p.add_argument('--n-groups',  default=10, type=int,
                   help='Use the first n groups of the series (0 = all)')
    p.add_argument('--N-resize',  default=56, type=int, help='Spatial resize for sens / images')

    # --- optimisation ---
    p.add_argument('--num-epochs',       default=101,  type=int)
    p.add_argument('--lr',               default=1e-1, type=float, help='Adam learning rate')
    p.add_argument('--lambda-reference', default=1e2,  type=float,
                   help='Weight of the L2 similarity between the estimates and the reference motion '
                        'linearly interpolated to all navigators (0 = off)')
    p.add_argument('--weighting', default='none', choices=['none', 'effective'],
                   help="Per-navigator data-term weights: 'none' = unweighted 1-|rho|; 'effective' = "
                        "w_n * 0.5 * (1-|rho_n|^2) with w_n from the effective SNR |rho_n|^2 / (1-|rho_n|^2), "
                        "re-estimated during the optimisation")

    # --- outputs ---
    p.add_argument('--save-every',     default=100,  type=int, help='Epochs between checkpoint saves')
    p.add_argument('--save-suffix',    default=None, help='Override auto-generated save suffix')
    p.add_argument('--results-name',   default=None,
                   help='File name of the saved trajectories in dir_matching (default: resMatch_<suffix>.pkl)')
    p.add_argument('--save-dir-fig',   default=None, help='Figure output directory')
    p.add_argument('--save-dir-match', default=None, help='Match results output directory')
    p.add_argument('--no-plots', action='store_true', help='Skip matplotlib plots')

    return p


def _parse_navis(spec: str) -> np.ndarray:
    # comma-separated indices and ranges, e.g. '0-12', '0,2,4' or '0,2-12'
    navis = []
    for item in spec.split(','):
        item = item.strip()
        if '-' in item:
            a, b = item.split('-')
            navis.extend(range(int(a), int(b) + 1))
        elif item:
            navis.append(int(item))
    return np.array(sorted(set(navis)), dtype=int)


def _in_notebook():
    # 'ipykernel' in sys.modules is not enough: sigpy.plot imports ipykernel
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        return False


args = _build_parser().parse_args([] if _in_notebook() else None)

# %%
# ---- parameters (edit for interactive runs) ----
input_mat  = args.input
date_tag   = args.date
series_ii  = args.series_ii

init             = args.init
do_svd           = args.do_svd
navis_to_use     = _parse_navis(args.navis_to_use) if args.navis_to_use else None   # None: from the .mat
N_GR_scan        = args.N_GR_scan
n_groups         = args.n_groups or N_GR_scan
N_resize         = args.N_resize

num_epochs       = args.num_epochs
lr               = args.lr
lambda_reference = args.lambda_reference
weighting        = args.weighting
save_every       = args.save_every

# 'effective' weighting: epochs between weight updates (weights = 1 before the first update),
# and clip range of the mean-1 normalised weights so that no navigator dominates
REWEIGHT_EVERY = 10
WEIGHT_CLIP    = (0.3, 3.0)

no_plots = args.no_plots

# %% [markdown]
# ### Device Setup

# %%
device_torch = torch.device(f'cuda:{args.gpu}')

cmpl_torch  = lambda x: torch.from_numpy(x).to(torch.cfloat).to(device_torch)
float_torch = lambda x: torch.from_numpy(x).to(torch.float).to(device_torch)

groups_slice = slice(N_GR_scan * series_ii, N_GR_scan * series_ii + min(n_groups, N_GR_scan))

# %% [markdown]
# ### Load Data

# %%
data = loadmat(input_mat)

coords_navi_c_tr = data['coords_at']
ksp_gnct         = data['ksp_gnct'][groups_slice]          # (N_GR, Nt_navi, N_ch, N_t)
motion_Est_gnp   = data['motion_Est_gnp'][groups_slice]    # (N_GR, Nt_navi, 6)
sens_dummy_cxyz  = data['sens_cxyz']
img_nxyz         = data['img_nxyz']
mo_GT_gp         = data['mo_GT_gp'][groups_slice]          # registration, sampled at the end of each group
u_H_nctt         = data.get('u_H_nctt', None)

if navis_to_use is None:
    # reliable navigators selected by main_motion_estimation.py, otherwise all
    navis_to_use = np.ravel(data['navis_used']).astype(int) if 'navis_used' in data \
                   else np.arange(data['ksp_gnct'].shape[1])
print(f'navigators optimised: {list(navis_to_use)}')

print(f'ksp_gnct       {ksp_gnct.shape}')
print(f'motion_Est_gnp {motion_Est_gnp.shape}')

# %% [markdown]
# ### Preprocessing

# %%
sens_dummy_cxyz = sp.resize(sens_dummy_cxyz, (sens_dummy_cxyz.shape[0], N_resize, N_resize, N_resize))
img_nxyz        = sp.resize(img_nxyz,        (img_nxyz.shape[0],        N_resize, N_resize, N_resize))

N_GR, Nt_navi_full, N_ch, N_t = ksp_gnct.shape

# %%
# SVD time-compression of navigator signals (matches main_motion_estimation convention)
if do_svd:
    if u_H_nctt is None:
        raise ValueError('do_svd=True but u_H_nctt not in .mat — re-export from main_motion_estimation with do_SVD_dictionary=True')
    ksp_compr_list = []
    for nii in navis_to_use:
        uH      = u_H_nctt[nii].transpose((0, 2, 1))          # (N_ch, N_compr, N_t)
        ksp_nii = np.matmul(uH[None], ksp_gnct[:, nii, ..., None]).squeeze(-1)
        ksp_compr_list.append(ksp_nii)
    ksp_gnct = np.stack(ksp_compr_list, axis=1)
    del ksp_compr_list
    print(f'SVD-compressed ksp {ksp_gnct.shape}')
else:
    ksp_gnct = ksp_gnct[:, navis_to_use]

# %% [markdown]
# ### Move to Torch

# %%
img_nxyz_t       = cmpl_torch(img_nxyz[navis_to_use])
sens_cxyz_t      = cmpl_torch(sens_dummy_cxyz)
ksp_gnct_t       = cmpl_torch(ksp_gnct)
coord_t          = float_torch(coords_navi_c_tr)
motion_Est_gnp_t = float_torch(motion_Est_gnp[:, navis_to_use, :])

uH_torch = None
if do_svd:
    uH_torch = cmpl_torch(u_H_nctt[navis_to_use].swapaxes(-1, -2))
    print(f'SVD basis {uH_torch.shape}')

N_GR, Nt_navi = motion_Est_gnp_t.shape[:2]
print(f'motion_Est_gnp_t {motion_Est_gnp_t.shape}')

# %% [markdown]
# ### Initial Trajectory (Dict-Match Estimate)

# %%
# dictionary-matching estimate and registration on all navigators (plots and saved results)
motion_Est_DM_flat = motion_Est_gnp.reshape(-1, 6)
motion_GT_flat     = np.tile(mo_GT_gp[:N_GR, None, :], (1, Nt_navi_full, 1)).reshape(-1, 6)

if not no_plots:
    plot_motion_traj(
        {'motion_Est': motion_Est_DM_flat, 'motion_GT': motion_GT_flat},
        img_resolution=4, N_GR=N_GR, Nt_navi=Nt_navi_full,
        save=False, figname='', translation_lims=[-2, 2], rotation_lims=[-3, 3],
        markers=['-o', '-o'],
    )

# unit conversion: voxels → mm, radians → degrees (forward model expects these units)
motion_Est_gnp_t[..., :3] *= 4.0
motion_Est_gnp_t[..., 3:] *= 180.0 / torch.pi

motion_reference_gp = mo_GT_gp.copy()
motion_reference_gp[..., :3] *= 4.0
motion_reference_gp[..., 3:] *= 180.0 / np.pi

# %% [markdown]
# ### Reference Motion for the Regularisation

# %%
# the reference is sampled at the last navigator of each group: interpolate linearly between the
# previous and the current group to every navigator, relative to the first group
reference_previous_gp = np.concatenate([motion_reference_gp[:1], motion_reference_gp[:-1]])
navigator_fraction    = ((navis_to_use + 1) / Nt_navi_full)[None, :, None]
motion_reference_gnp  = (reference_previous_gp[:, None]
                         + navigator_fraction * (motion_reference_gp - reference_previous_gp)[:, None]
                         - motion_reference_gp[:1, None])
motion_reference_gnp  = torch.tensor(motion_reference_gnp, dtype=torch.float, device=device_torch)

# %% [markdown]
# ### Optimisation Setup

# %%
if init == 'zero':
    motion_optim_gnp = nn.Parameter(torch.zeros_like(motion_Est_gnp_t), requires_grad=True)
else:
    motion_optim_gnp = nn.Parameter(motion_Est_gnp_t.clone(), requires_grad=True)
print(f'initializing motion from: {init}')

optimizer = optim.Adam([motion_optim_gnp], lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-3)
model = Forward_Motion(img_nxyz_t, sens_cxyz_t, coord_t, motion_optim_gnp).to(device_torch)

# %% [markdown]
# ### Output Paths

# %%
with open('/local_mount/space/mayday/data/users/aizadan/dictgator/data_info.json', 'r') as fh:
    metadata = json.load(fh)[date_tag][args.dataset]

save_suffix  = args.save_suffix or \
    f'optim_init{init}_ser{series_ii}_ngr{min(n_groups, N_GR_scan)}_lr{lr:.0e}_reference{lambda_reference:.0e}' \
    + ('' if weighting == 'none' else f'_w{weighting}')
dir_figure   = args.save_dir_fig   or os.path.join(metadata['dir_figures'], 'mo_estim')
dir_matching = args.save_dir_match or metadata['dir_matching']

os.makedirs(dir_figure,   exist_ok=True)
os.makedirs(dir_matching, exist_ok=True)


def interpolate_navigators(motion_gnp):
    """Optimised navigators (N_GR, len(navis_to_use), 6) → all navigators (N_GR, Nt_navi_full, 6).
    Navigators that were not optimised are linearly interpolated from their neighbours within the group."""
    all_navis = np.arange(Nt_navi_full)
    motion_full_gnp = np.zeros((motion_gnp.shape[0], Nt_navi_full, 6))
    for g_ii in range(motion_gnp.shape[0]):
        for p_ii in range(6):
            motion_full_gnp[g_ii, :, p_ii] = np.interp(all_navis, navis_to_use, motion_gnp[g_ii, :, p_ii])
    return motion_full_gnp


def save_results(tag=''):
    # undo unit scaling → back to voxels / radians, on a copy of the parameter being optimised
    motion_optim_final = motion_optim_gnp.detach().clone()
    motion_optim_final[..., :3] /= 4.0
    motion_optim_final[..., 3:] *= torch.pi / 180.0

    motion_Est_optim = interpolate_navigators(motion_optim_final.cpu().numpy())        # (N_GR, Nt_navi_full, 6)
    res_match = {
        'motion_Est_DM':    motion_Est_DM_flat.reshape(N_GR, Nt_navi_full, 6),
        'motion_Est_optim': motion_Est_optim,
        'motion_smooth_GT': motion_Est_optim.reshape(-1, 6),   # same, flat (N_GR * Nt_navi_full, 6), as read by plot_motion_traj
        'motion_GT':        motion_GT_flat,
        'navis_used':       navis_to_use,
        'weighting':        weighting,
        'lambda_reference': lambda_reference,
        'weights_history_n': np.array(weights_history),   # per update, over navis_used
    }
    results_name = args.results_name.removesuffix('.pkl') if args.results_name else f'resMatch_{save_suffix}'
    dict_name_to_save = os.path.join(dir_matching, f'{results_name}{tag}.pkl')
    with open(dict_name_to_save, 'wb') as fh:
        pickle.dump(res_match, fh)
    print(f'Results saved → {dict_name_to_save}')
    return res_match

# %% [markdown]
# ### Training Loop

# %%
def normalize_weights(snr_n):
    """mean-1 weights (keeps lambda_reference meaningful), clipped so that no navigator dominates"""
    weights_n = snr_n / max(snr_n.mean(), 1e-12)
    return np.clip(weights_n, *WEIGHT_CLIP)


losses, regul_losses = [], []
weights_n       = np.ones(Nt_navi)
weights_history = [weights_n.copy()]
rho2_gn         = np.zeros((N_GR, Nt_navi))
tic = time.perf_counter()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    data_loss = torch.tensor(0.0, device=device_torch)

    # effective SNR from the current fit (includes model mismatch)
    if weighting == 'effective' and epoch > 0 and epoch % REWEIGHT_EVERY == 0:
        rho2_n    = np.clip(rho2_gn.mean(axis=0), 0.0, 1.0 - 1e-6)
        weights_n = normalize_weights(rho2_n / (1.0 - rho2_n))
        weights_history.append(weights_n.copy())
        print(f'epoch {epoch} effective weights:', np.array2string(weights_n, precision=2))

    for g_ii in range(N_GR):
        for nii in range(Nt_navi):
            predicted = model._forward_op(
                motion_optim_gnp[g_ii, nii].reshape(1, 1, 6),
                img_nxyz_t[[nii]],
                coord_t,
                uH=uH_torch[nii] if uH_torch is not None else None,
            )
            loss = complex_dot_loss(predicted, ksp_gnct_t[[g_ii]][:, [nii]])
            if weighting == 'none':
                data_loss += loss
            else:
                # complex scale profiled out; 0.5 keeps the scale of 1-|rho| near rho = 1
                rho = 1.0 - loss
                rho2_gn[g_ii, nii] = float(rho.detach() ** 2)
                data_loss += float(weights_n[nii]) * 0.5 * (1.0 - rho ** 2)

    # estimates and reference have different reference points: remove the constant offset before the L2
    residual   = motion_optim_gnp - motion_reference_gnp
    motion_reg = lambda_reference * ((residual - residual.mean(dim=(0, 1), keepdim=True)) ** 2).mean()

    data_loss  = data_loss / (N_GR * Nt_navi)
    total_loss = data_loss + motion_reg
    total_loss.backward()
    optimizer.step()

    losses.append(float(total_loss.detach().cpu()))
    regul_losses.append(float(motion_reg.detach().cpu()))
    scheduler.step(float(total_loss))

    if epoch % 50 == 0:
        print(f'Epoch {epoch:4d}/{num_epochs}  '
              f'total {total_loss.item():.3e}  '
              f'data {data_loss.item():.3e}  '
              f'reg {motion_reg.item():.3e}')

    if (epoch + 1) % save_every == 0:
        save_results(tag=f'_ep{epoch + 1}')

print(f'\nDone in {time.perf_counter()-tic:.1f}s')

# %% [markdown]
# ### Save & Plot

# %%
if not no_plots:
    plt.figure()
    plt.plot(losses,       label='Total')
    plt.plot(regul_losses, label='Regularisation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

res_match = save_results()

# %%
Nt_plot = min(13, Nt_navi_full)

def _select(x):
    return x.reshape(N_GR, Nt_navi_full, 6)[:, :Nt_plot, :].reshape(-1, 6)

if not no_plots:
    plot_motion_traj(
        {
            'motion_Est':       _select(res_match['motion_Est_DM'].reshape(-1, 6)),
            'motion_GT':        _select(motion_GT_flat),
            'motion_smooth_GT': _select(res_match['motion_Est_optim'].reshape(-1, 6)),
        },
        img_resolution=4, N_GR=N_GR, Nt_navi=Nt_plot,
        save=False,
        figname=os.path.join(dir_figure, f'motion_{save_suffix}'),
        translation_lims=[-2, 2], rotation_lims=[-2, 2],
        plot_Est=True, plot_GT=True, plot_smooth_GT=True, plot_smooth_noGT=False,
        figsize=(15, 6),
        markers=['-o', '-o', '-', '-'],
        labels_methods=['DM', 'Registration', 'DM, refined'],
    )
    plt.show()
