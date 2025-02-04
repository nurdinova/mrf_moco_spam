import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import sigpy as sp
import sigpy.plot as pl
from motion_ops_gpu import Translation_Op, Rotation_Op, generate_grids, Rigid_Motion
import time 
from plot_utils import plot_3dimg
from motion_estimation import compress_dictionary_chunk

def dict_gen(input_params, coils_before_motion=False, debug=0):
    """
    Function to simulate the Nav acquisition
    y = nuFFT * S * R_phi * input
    if coils_before_motion:
        y = nuFFT * R_phi * S * input
    
    Warning: dimensions Nm, N_chunks, Nc, Nx, Ny, Nz need to be even 
            tom match bart nufft
    
    input_params : dict
        contains dict_gen settings
    
    input : ndarray (n_0, .., n_dim-1)
        input image needs to be cubic!
    motion : dict
        parameters for rigid motion in 6 DOF
        (dx, dy, dz, theta_x, theta_y, theta_z)
    sens : ndarray (n_0, n_1, N_channels)
    k_coords : ndarray (...,n_dim)
        should be scaled to have range between -n // 2 and n // 2
    sigma : float 
        Gaussian noise
    rot_center : string
        opitions = edge, center
        
    Returns:
        k-space Nav signal at the acquired k-points
    """
    
    in_img_qxyz = input_params["input_img_xyzq"].transpose((-1, 0, 1, 2))
    if 'device' in input_params.keys(): 
        device = input_params['device']
    else:
        device = sp.get_device(in_img_qxyz)
    mvd = lambda x : sp.to_device(x, device)
    mvc = lambda x : sp.to_device(x, sp.Device(-1))
    xp = device.xp
    
    sens_gpu_cxyz = np.transpose(input_params["sens_xyzc"], axes=(3, 0, 1, 2))
    N_ch = sens_gpu_cxyz.shape[0]

    k_crds_gpu = input_params["k_coords_ka"]
    N_t = k_crds_gpu.shape[0]
    
    rot_center = input_params["rot_center"]
    
    if "motion_mp" not in input_params:
        motion_mp = np.stack(np.meshgrid(input_params["shifts_x"], input_params["shifts_y"], \
                                             input_params["shifts_z"], input_params["angles_x"], \
                                             input_params["angles_y"], input_params["angles_z"]), -1).reshape((-1, 6))
    else:
        motion_mp = sp.to_device(input_params["motion_mp"], sp.Device(-1))
    n_pars = motion_mp.shape[0]
        
    with device:
        if sp.get_device(sens_gpu_cxyz) != device:
            sens_gpu_cxyz = mvd(sens_gpu_cxyz)
        if sp.get_device(k_crds_gpu) != device:
            k_crds_gpu = mvd(k_crds_gpu)
            
    imshape_qxyz = in_img_qxyz.shape
    N = np.max(imshape_qxyz)
    Nq = imshape_qxyz[0]
    in_img = sp.resize(in_img_qxyz, [Nq, N, N, N]) # motion operator requires cubic volume
        
    if "v_ct_tnew" in input_params:
        do_compress_dict = True
        u_t_tnew = input_params["v_ct_tnew"]
        N_ch_new, _, N_t_new = u_t_tnew.shape
    else:
        do_compress_dict = False
        N_ch_new = N_ch
        N_t_new = N_t
        u_t_tnew = None
        
    if "Phi_qnr" in input_params:
        do_unroll_contrast = True  
        Phi_qnr = mvd(input_params["Phi_qnr"])
        Nq_new = Phi_qnr.shape[-2]
    else:
        do_unroll_contrast = False
        Nq_new = Nq
        Phi_qnr = None

    # allocate memmap for the dict with the provided name and path 
    dict_mo_mctq = np.memmap( input_params["dict_fullname"]+'.dat', dtype=np.complex64,
              mode='w+', shape=(n_pars, N_ch_new, N_t_new, Nq_new))
    dict_mo_idx = np.memmap( input_params["dict_fullname"]+'_moIdx.dat', dtype=np.float32,
              mode='w+', shape=(n_pars, 6))
    dict_mo_idx[:] = motion_mp[:]
    dict_mo_idx.flush()
    
    motion_mp = mvd(motion_mp)

    if coils_before_motion:
        imshape_cqxyz = np.insert(imshape_qxyz, 0, N_ch)
        print("Generating dictionary y = nuFFT * R_phi * S * input..........")
    else:
        imshape_cqxyz = np.insert(imshape_qxyz, 0, 1)
        print("Generating dictionary y = nuFFT * S * R_phi * input..........")
        
    if rot_center == "edge":
        img_2pad = xp.zeros((Nq, 2*N, 2*N, 2*N), complex)
        img_2pad[:, :N, N:, N:] = in_img
        img_pad_gpu = img_2pad
    else:
        img_pad_gpu = in_img
        
    tic = time.perf_counter()
    with device: 
        img_pad_gpu_c_qxyz = img_pad_gpu[None,...]
        if debug:
            print('Padded image shape ', img_pad_gpu_c_qxyz.shape)

        chunksize = 4
        N_chunks = (n_pars + chunksize - 1) // chunksize        

        if coils_before_motion:
#             print(f'img shape {img_pad_gpu_c_xyz.shape} ')
            img_pad_gpu_c_qxyz = sens_gpu_cxyz[:, None, ...] * img_pad_gpu_c_qxyz
            
        for chunk_ii in range(N_chunks):
            start_idx = chunk_ii * chunksize
            end_idx = min(n_pars, (chunk_ii + 1) * chunksize)
            
            Transl_Rot_Op = Rigid_Motion(img_pad_gpu_c_qxyz.shape, motion_mp[start_idx:end_idx])
            Mx_cqmxyz = Transl_Rot_Op._apply(img_pad_gpu_c_qxyz)
            
            if rot_center == "edge":
                Mx_cqmxyz = Mx_cqmxyz[..., :N, N:, N:]
            if debug:
                print(f'after motion {Mx_cqmxyz.shape}')
                pl.ImagePlot(Mx_cqmxyz[0, 0, :, N//2, :, :], z=0, y=-1, x=-2, title='motion states')

            if not coils_before_motion:
                SMx_cqmxyz = sens_gpu_cxyz[:, None, None, ...] * (Mx_cqmxyz) 
            else:
                SMx_cqmxyz = Mx_cqmxyz
            del Mx_cqmxyz
                
            if debug:
                print(f'before nufft SMx_cqmxyz {SMx_cqmxyz.shape}')
                pl.ImagePlot(SMx_cqmxyz[:, 0, 0, N//2, :, :], z=0, y=-1, x=-2, title='coils')
                
            # nufft input (channel, n_dim-1, .., n_1, n_0)
            dict_full_mctq = sp.nufft(input=SMx_cqmxyz, coord=k_crds_gpu).transpose((2, 0, 3, 1))   
            if 'noise_sigma' in input_params:
                noise_sigma = input_params['noise_sigma']
                dict_full_mctq = dict_full_mctq + xp.random.normal(scale=(noise_sigma \
                                 * xp.max(dict_full_mctq.real)), size=dict_full_mctq.shape) \
                                 + 1j * xp.random.normal(scale=(noise_sigma \
                                 * xp.max(dict_full_mctq.imag)), size=dict_full_mctq.shape) 
                
            if do_compress_dict or do_unroll_contrast:
                dict_full_mctq = compress_dictionary_chunk(dict_full_mctq, Phi_qnr, u_t_tnew)
            
            dict_mo_mctq[start_idx:end_idx] = mvc(dict_full_mctq)
                
            if debug:
                toc = time.perf_counter()
                print(f"time = {toc - tic:0.4f} seconds, chunk %d / %d"%(chunk_ii, N_chunks))   
                
    dict_mo_mctq.flush()
    toc = time.perf_counter()
    print(f"Finished! Total time = {toc - tic:0.4f} seconds")
    return dict_mo_mctq, dict_mo_idx

def dict_gen_ksp(input_params):
    """
    Motion simulations in k-space
    img_cxyz : multi-coil dummy image recon at the fixed navigator contrast
    
    """
    img_xyzq = input_params['input_img_xyzq']
    crds_ka = input_params['k_coords_ka'] 
    motion_mp = input_params['motion_mp']
    sens_cxyz = input_params['sens_xyzc'].transpose((-1, 0, 1, 2))
    n_pars = motion_mp.shape[0]
    N, _, _, nq = img_xyzq.shape
    nc = sens_cxyz.shape[0]
    nt = crds_ka.shape[0]
    
    if "v_ct_tnew" in input_params:
        do_compress_dict = True
        u_t_tnew = input_params["v_ct_tnew"]
        nc_new, _, nt_new = u_t_tnew.shape
    else:
        do_compress_dict = False
        nc_new = N_ch
        nt_new = N_t
        u_t_tnew = None
        
    device = sp.get_device(img_xyzq)
    xp = device.xp
    
    dict_mo_mctq = np.memmap(input_params["dict_fullname"]+'.dat', dtype=np.complex64,
              mode='w+', shape=(n_pars, nc_new, nt_new, nq))
    dict_mo_idx = np.memmap(input_params["dict_fullname"]+'_moIdx.dat', dtype=np.float32,
              mode='w+', shape=(n_pars, 6))
    dict_mo_idx[:] = motion_mp.get()
    dict_mo_idx.flush()

    with device:
        if sp.get_device(motion_mp) != device:
            motion_mp = mvd(motion_mp)
        K_range = xp.max(crds_ka) - xp.min(crds_ka)
        crds_ka = crds_ka / K_range
        img_cqxyz = (sens_cxyz[..., None] * img_xyzq[None]).transpose((0, -1, 1, 2, 3))
        for m_ii in range(n_pars):
                # smth is off with cupy stacking for Rx
                dx, dy, dz, phi_x, phi_y, phi_z = motion_mp[m_ii, :]
#                 phi_x = -phi_x
#                 phi_y = -phi_y
#                 dx = -dx

                Rx = xp.stack([
                    xp.array([1, 0, 0]), 
                    xp.array([xp.array(0), xp.cos(phi_x), -xp.sin(phi_x)]), 
                    xp.array([xp.array(0), xp.sin(phi_x), xp.cos(phi_x)])   
                ])
                Ry = xp.stack([
                    xp.array([xp.cos(phi_y), xp.array(0), -xp.sin(phi_y)]), 
                    xp.array([0, 1, 0]), 
                    xp.array([xp.sin(phi_y), xp.array(0), xp.cos(phi_y)])
                ])
                Rz = xp.stack([
                    xp.array([xp.cos(phi_z), -xp.sin(phi_z), xp.array(0)]), 
                    xp.array([xp.sin(phi_z), xp.cos(phi_z), xp.array(0)]),
                    xp.array([0, 0, 1])
                ])

                # crds are in [-0.5, 0.5]
                crds_ak_rot = xp.matmul(Rz, crds_ka.T)
                crds_ak_rot = xp.matmul(Ry, crds_ak_rot)
                crds_ak_rot = xp.matmul(Rx, crds_ak_rot)

                ph_ramp_x_k = xp.exp( -2j * xp.pi * dx * crds_ak_rot[0] / N )
                ph_ramp_y_k = xp.exp( -2j * xp.pi * dy * crds_ak_rot[1] / N )
                ph_ramp_z_k = xp.exp( -2j * xp.pi * dz * crds_ak_rot[2] / N )
                ph_ramp_xyz_ckq = (ph_ramp_x_k * ph_ramp_y_k * ph_ramp_z_k)[None, :, None] 
                
                signal_ctq = sp.nufft(input=img_cqxyz, coord=crds_ak_rot.T*K_range, oversamp=4, width=12).transpose((0, 2, 1)) * ph_ramp_xyz_ckq
                if do_compress_dict:
                    signal_ctq = compress_dictionary_chunk(signal_ctq[None], None, u_t_tnew)
                dict_mo_mctq[m_ii] = signal_ctq.get() 
    dict_mo_mctq.flush()
    return dict_mo_mctq, dict_mo_idx