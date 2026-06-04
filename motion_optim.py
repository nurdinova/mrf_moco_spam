#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ### Autodiff-based motion refinement of dictionary-matching estimates
#
# Loads the `.mat` exported by `MoEstim_Invivo`, runs a PyTorch forward-model
# optimisation (k-space data consistency + L1 temporal regularisation), and saves
# refined motion trajectories.
#
# | Mode | `--init-zero` | Description |
# |------|---------------|-------------|
# | Refinement (default) | off | initialise from dict-match estimate |
# | Baseline | on | initialise from zero; for comparison |
#
# Run as a notebook: edit the parameter cell below.
# Run as a script:   `python Motion_optim_GT.py --input data.mat --date 250425 ...`

# %% [markdown]
# ### Imports

# %%
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

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
from src.torch_optim import Forward_Motion, motion_regularization_l1, complex_dot_loss

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
# Run as a script:   `python Motion_optim_GT.py --input <path> --date 250425 ...`

# %%
_DESCRIPTION = (
    "Gradient-based motion refinement of dictionary-matching estimates.\n\n"
    "Loads a .mat exported by MoEstim_Invivo, runs a PyTorch forward-model\n"
    "optimisation (data consistency + L1 regularisation), and saves refined\n"
    "motion trajectories.\n\n"
    "Initialisation: dict-match estimate by default; use --init-zero for a\n"
    "zero-initialised baseline comparison (reflected in the save suffix)."
)


def _build_parser():
    p = argparse.ArgumentParser(
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- input ---
    p.add_argument('--input',     default=None, type=str,
                   help='Path to .mat exported by MoEstim_Invivo')
    p.add_argument('--date',      default='250425', type=str,
                   help='Scan date tag (used in output naming)')
    p.add_argument('--series-ii', default=0, type=int,
                   help='Series index: selects groups [N_GR_scan*i : N_GR_scan*(i+1)]')

    # --- compute ---
    p.add_argument('--gpu', default=2, type=int, help='CUDA device index')

    # --- initialisation ---
    p.add_argument('--init-zero', action='store_true',
                   help='Init from zero (baseline); default: from dict-match estimate')

    # --- data ---
    p.add_argument('--no-svd', action='store_false', dest='do_svd',
                   help='Skip SVD time-compression of navigator signals')
    p.set_defaults(do_svd=True)
    p.add_argument('--navis-to-use', default='0-12', type=str,
                   help="Navigator indices, e.g. '0-12' or '0,2,4'")
    p.add_argument('--N-GR-scan', default=36, type=int, help='Groups per acquisition series')
    p.add_argument('--N-resize',  default=56, type=int, help='Spatial resize for sens / images')

    # --- optimisation ---
    p.add_argument('--num-epochs',  default=301,  type=int)
    p.add_argument('--lambda-reg',  default=9e-2, type=float, help='L1 motion regularisation weight')
    p.add_argument('--lr',          default=1e-1, type=float, help='Adam learning rate')

    # --- forward model ---
    p.add_argument('--n-time-segments',     default=2,   type=int)
    p.add_argument('--readout-duration-ms', default=7.0, type=float)
    p.add_argument('--number-readouts',     default=3,   type=int)

    # --- outputs ---
    p.add_argument('--save-suffix',    default=None, help='Override auto-generated save suffix')
    p.add_argument('--save-dir-fig',   default=None, help='Figure output directory')
    p.add_argument('--save-dir-match', default=None, help='Match results output directory')
    p.add_argument('--no-plots', action='store_true', help='Skip matplotlib plots')

    return p


def _parse_navis(spec: str) -> np.ndarray:
    spec = spec.strip()
    if '-' in spec and ',' not in spec:
        a, b = spec.split('-')
        return np.arange(int(a), int(b) + 1)
    return np.array([int(x) for x in spec.split(',') if x], dtype=int)


args = _build_parser().parse_args([] if 'ipykernel' in sys.modules else None)

# %%
# ---- parameters (edit for interactive runs) ----
input_mat  = args.input   or './data_final_pinv_SVD_DictMatch_ser4_v2_noKalman_smallDict'
date_tag   = args.date
series_ii  = args.series_ii

init_zero        = False # args.init_zero
do_svd           = args.do_svd
navis_to_use     = [12] # _parse_navis(args.navis_to_use)
N_GR_scan        = args.N_GR_scan
N_resize         = args.N_resize

num_epochs  = 2 # args.num_epochs
lambda_reg  = args.lambda_reg
lr          = args.lr

n_time_segments     = args.n_time_segments
readout_duration_ms = args.readout_duration_ms
number_readouts     = args.number_readouts

no_plots = args.no_plots

navi_to_interpolate = []

# %% [markdown]
# ### Device Setup

# %%
device_torch = torch.device(f'cuda:{args.gpu}')
device_sp    = sp.Device(args.gpu)

cmpl_torch  = lambda x: torch.from_numpy(x).to(torch.cfloat).to(device_torch)
float_torch = lambda x: torch.from_numpy(x).to(torch.float).to(device_torch)

groups_slice = slice(N_GR_scan * series_ii, N_GR_scan * (series_ii + 1))

# %% [markdown]
# ### Load Data

# %%
data = loadmat(input_mat)

coords_navi_c_tr = data['coords_at']
ksp_gnct         = data['ksp_gnct'][groups_slice]          # (N_GR, Nt_navi, N_ch, N_t)
motion_Est_gnp   = data['motion_Est_gnp'][groups_slice]    # (N_GR, Nt_navi, 6)
sens_dummy_cxyz  = data['sens_cxyz']
img_nxyz         = data['img_nxyz']
mo_GT_gp         = data['mo_GT_gp'][groups_slice]
u_H_nctt         = data.get('u_H_nctt', None)

print(f'ksp_gnct       {ksp_gnct.shape}')
print(f'motion_Est_gnp {motion_Est_gnp.shape}')

# %% [markdown]
# ### Preprocessing

# %%
sens_dummy_cxyz = sp.resize(sens_dummy_cxyz, (sens_dummy_cxyz.shape[0], N_resize, N_resize, N_resize))
img_nxyz        = sp.resize(img_nxyz,        (img_nxyz.shape[0],        N_resize, N_resize, N_resize))

N_GR, Nt_navi_full, N_ch, N_t = ksp_gnct.shape

# %%
# SVD time-compression of navigator signals (matches MoEstim_Invivo convention)
if do_svd:
    if u_H_nctt is None:
        raise ValueError('do_svd=True but u_H_nctt not in .mat — re-export from MoEstim_Invivo with do_SVD_dictionary=True')
    ksp_compr_list = []
    for nii in navis_to_use:
        uH      = u_H_nctt[nii].transpose((0, 2, 1))          # (N_ch, N_compr, N_t)
        ksp_nii = np.matmul(uH[None], ksp_gnct[:, nii, ..., None]).squeeze(-1)
        ksp_compr_list.append(ksp_nii)
    ksp_gnct = np.stack(ksp_compr_list, axis=1)
    del ksp_compr_list
    print(f'SVD-compressed ksp {ksp_gnct.shape}')

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
motion_Est_DM_flat = motion_Est_gnp_t.cpu().detach().reshape(-1, 6).numpy()
motion_GT_flat     = np.tile(mo_GT_gp[:N_GR, None, :], (1, Nt_navi, 1)).reshape(-1, 6)

if not no_plots:
    plot_motion_traj(
        {'motion_Est': motion_Est_DM_flat, 'motion_GT': motion_GT_flat},
        img_resolution=4, N_GR=N_GR, Nt_navi=Nt_navi,
        save=False, figname='', translation_lims=[-2, 2], rotation_lims=[-3, 3],
        markers=['-o', '-o'],
    )

# unit conversion: voxels → mm, radians → degrees (forward model expects these units)
motion_Est_gnp_t[..., :3] *= 4.0
motion_Est_gnp_t[..., 3:] *= 180.0 / torch.pi

mo_GT_scaled = mo_GT_gp.copy()
mo_GT_scaled[..., :3] *= 4.0
mo_GT_scaled[..., 3:] *= 180.0 / np.pi

# %% [markdown]
# ### Optimisation Setup

# %%
motion_GT_tensor = torch.tensor(mo_GT_scaled[:, None, :], device=device_torch)
n_dB0_bases      = 9

# initialisation: dict-match estimate (refinement) or zero (baseline)
init_tag = 'initZero' if init_zero else 'initDM'
if init_zero:
    motion_optim_gnp = nn.Parameter(torch.zeros_like(motion_Est_gnp_t), requires_grad=True)
else:
    motion_optim_gnp = nn.Parameter(motion_Est_gnp_t.clone(), requires_grad=True)

# dB0 spherical-harmonic coefficients (kept fixed — set requires_grad=True to train)
beta_coef_gnb = nn.Parameter(
    torch.zeros((N_GR, Nt_navi, n_dB0_bases), dtype=torch.float, device=device_torch),
    requires_grad=False,
)

optimizer = optim.Adam([motion_optim_gnp], lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                              min_lr=1e-3, verbose=True)
model = Forward_Motion(img_nxyz_t, sens_cxyz_t, coord_t, motion_optim_gnp).to(device_torch)

# %% [markdown]
# ### Training Loop

# %%
losses, regul_losses = [], []
tic = time.perf_counter()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    data_loss = torch.tensor(0.0, device=device_torch)

    for g_ii in range(N_GR):
        for nii, navi_idx in enumerate(navis_to_use):
            predicted = model._forward_dB0_motion(
                motion_optim_gnp[g_ii, nii].reshape(1, 1, 6),
                beta_coef_gnb[g_ii, nii],
                img_nxyz_t[[nii]],
                n_time_segments=n_time_segments,
                readout_duration_ms=readout_duration_ms,
                number_readouts=number_readouts,
                uH=uH_torch[nii] if uH_torch is not None else None,
            )
            data_loss += complex_dot_loss(predicted, ksp_gnct_t[[g_ii]][:, [nii]])

    # interpolate low-SNR navigator 0 from its neighbour
    with torch.no_grad():
        for nii_interp in navi_to_interpolate:
            closest_point = min(Nt_navi, nii_interp + 1)
            motion_optim_gnp[:, nii_interp, :].copy_(motion_optim_gnp[:, closest_point, :])

    motion_reg = motion_regularization_l1(
        torch.cat([motion_optim_gnp, motion_GT_tensor], dim=1).reshape(-1, 6),
        beta=lambda_reg,
    )
    data_loss  = data_loss / (N_GR * Nt_navi)
    total_loss = data_loss + motion_reg
    total_loss.backward()
    optimizer.step()

    losses.append(float(total_loss.detach().cpu()))
    regul_losses.append(float(motion_reg.detach().cpu()))
    scheduler.step(float(data_loss))

    if epoch % 50 == 0:
        print(f'Epoch {epoch:4d}/{num_epochs}  '
              f'total {total_loss.item():.3e}  '
              f'data {data_loss.item():.3e}  '
              f'reg {motion_reg.item():.3e}')

print(f'\nDone in {time.perf_counter()-tic:.1f}s')

# %% [markdown]
# ### Results

# %%
if not no_plots:
    plt.figure()
    plt.plot(losses,       label='Total')
    plt.plot(regul_losses, label='Regularisation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

# undo unit scaling → back to voxels / radians
motion_optim_final = motion_optim_gnp.detach().clone()
motion_optim_final[..., :3] /= 4.0
motion_optim_final[..., 3:] *= torch.pi / 180.0
motion_optim_np = motion_optim_final.cpu().numpy()   # (N_GR, Nt_navi, 6)

# %% [markdown]
# ### Save & Plot

# %%
input_dir    = Path(input_mat).parent
save_suffix  = args.save_suffix or f'optim_{init_tag}_ser{series_ii}_lr{lr:.0e}_reg{lambda_reg:.0e}'
dir_figure   = args.save_dir_fig   or str(input_dir / 'figures')
dir_matching = args.save_dir_match or str(input_dir / 'match_results')

os.makedirs(dir_figure,   exist_ok=True)
os.makedirs(dir_matching, exist_ok=True)

res_match = {
    'motion_Est_DM':    motion_Est_DM_flat.reshape(N_GR, Nt_navi, 6),
    'motion_Est_optim': motion_optim_np,
    'motion_GT':        motion_GT_flat,
}
dict_name_to_save = os.path.join(dir_matching, f'resMatch_{save_suffix}.pkl')
with open(dict_name_to_save, 'wb') as fh:
    pickle.dump(res_match, fh)
print(f'Results saved → {dict_name_to_save}')

# %%
Nt_plot = min(13, Nt_navi)

def _select(x):
    return x.reshape(N_GR, Nt_navi, 6)[:, :Nt_plot, :].reshape(-1, 6)

if not no_plots:
    plot_motion_traj(
        {
            'motion_Est':       _select(res_match['motion_Est_DM'].reshape(-1, 6)),
            'motion_GT':        _select(motion_GT_flat),
            'motion_smooth_GT': _select(motion_optim_np.reshape(-1, 6)),
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
