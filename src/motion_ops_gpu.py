import sigpy as sp
from sigpy.linop import Linop 
import copy
import numpy as np

import warnings
from sigpy.linop import Linop

def generate_grids(Nkm, xp=np):
    """
    Generate spatio, temporal and spatio-temporal grids for cubic image of shape Nkm.
    
    Nkm: int
        EVEN number indicating the size of each input dimension
    
    """
    min_k = -np.floor(Nkm/2) 
    max_k = np.ceil(Nkm/2) 
    Nk = max_k - min_k
    
    r_m = xp.arange(min_k, max_k, 1)
    kGrid = 2 / Nk * np.pi * r_m 
    rGrid = r_m     
    rkGrid = xp.outer(rGrid, kGrid)
    
    return rGrid, kGrid, rkGrid

class Rotation_Op(Linop):
    """
    Sinc-interpolated 3D rigid transforms, operating in k-space.
    
    img : 4d-array and 3d-array
        dimensions cxyz or xyz - coil and spatial
        image, that will be tranformed around the axes in the center of the plane
        give 4x padded image if want to compare to FLIRT rotations
    traj : ndarray 
        of size (N_mot_samples, 7)
        rigid motion trajectory [time, dx, dy, dz, theta_x, theta_y, theta_z]
        motion will be interpolated at the sequence sampling points if interp_motion is True
    Nkm : int
        size of the image
        
    Return 
        image_tr : 3d-array
        Warped image
    """

    def __init__(self, ishape=None, motion_mp=None, pad=True, device=None):
        """
        Input image needs to be 2D or 3D cube with dimension N. 
        """
        warnings.warn('Use Rigid_Motion operator please')
        if not pad or device is not None:
            warnings.warn('pad and device args depreciated, please remove')
        if ishape is None or motion_mp is None:
            raise ValueError('Please provide motion and image shape in Rotation._init_')
            
        assert (ishape[-1] == ishape[-2] == ishape[-3]), 'Input image should be cubic'
            
        self.device = sp.get_device(motion_mp)
        self.xp = self.device.xp
        
        self.ishape = ishape
        self.oshape = tuple([motion_mp.shape[0]] + list(ishape))
        self.mot_traj = motion_mp
        with self.device:
            _, _, rkGrid = generate_grids(ishape[-1], self.xp)
                
        # prep shear matrices 
        self.do_rot = False
        with self.device:
            thetas_mp = self.mot_traj[:, -3:] * self.xp.array([-1, 1, 1]).reshape((1, 3))

            if (thetas_mp.any()):
                self.do_rot = True

            if self.do_rot:
                tantheta2_mp = 1j * self.xp.tan(thetas_mp / 2)
                sintheta_mp = -1j * self.xp.sin(thetas_mp)
                
                # 1. Rotations
                V_mat_tan_mrk = [] # of size 3 - for each rotation axis,
                                   # each element has size (rkGrid_size)
                V_mat_sin_mrk = []
                for ax_ in range(3):
                    exp_mrk = self.xp.exp((tantheta2_mp[:, [2-ax_], None] * rkGrid[None, ...])) 
                    V_mat_tan_mrk.append(exp_mrk)

                    exp_mrk = self.xp.exp((sintheta_mp[:, [2-ax_], None] * rkGrid[None, ...]))
                    V_mat_sin_mrk.append(exp_mrk)  
                self.V_mat_tan_mrk = V_mat_tan_mrk
                self.V_mat_sin_mrk = V_mat_sin_mrk
        
        super().__init__(self.oshape, self.ishape)
    
    def _reset_motion_params(self, motion_params):
        """
        motion_params - vector of shape(6,) - translations and rotations
            rotation angles are in [rads]
        """
        raise AttributeError('Class was changed to init with motion')
    

    def _apply(self, input):
        """
        input is brxyz
        """
        if (sp.get_device(input) != self.device):
            raise ResourceWarning(f'Input is on {sp.get_device(input)}, while the current is {self.device}')
            input = sp.to_device(input, self.device)
            
        with self.device:            

            if input.ndim == 3:
                # xyz
                image_brxyz = xp.expand_dims(input, axis=(0, 1))
            elif input.ndim >= 4:
                # bxyz
                image_brxyz = xp.expand_dims(input, axis=(-4))
                     
            if self.do_rot:
                per = [[-3, -1, -2], [-2, -3, -1]]
                for axis in [0, 1, 2]:
                    newaxis = 2 - axis + 1
                    V_tan_exp_brxyz = self.xp.expand_dims(self.V_mat_tan_mrk[axis], axis=(newaxis))
                    V_sin_exp_brxyz = self.xp.expand_dims(self.V_mat_sin_mrk[axis], axis=(newaxis))

                    image_brxyz = self.xp.fft.fftshift(self.xp.fft.fft(image_brxyz, axis=per[0][axis]), axes=per[0][axis])
                    image_brxyz =  xp.multiply(V_tan_exp_brxyz, image_brxyz) 
                    image_brxyz = self.xp.fft.ifft(self.xp.fft.ifftshift(image_brxyz, axes=per[0][axis]), axis=per[0][axis])

                    image_brxyz = self.xp.fft.fftshift(self.xp.fft.fft(image_brxyz, axis=per[1][axis]), axes=per[1][axis])
                    image_brxyz = xp.multiply(V_sin_exp_brxyz, image_brxyz)
                    image_brxyz = self.xp.fft.ifft(self.xp.fft.ifftshift(image_brxyz, axes=per[1][axis]), axis=per[1][axis])

                    image_brxyz = self.xp.fft.fftshift(self.xp.fft.fft(image_brxyz, axis=per[0][axis]), axes=per[0][axis])
                    image_brxyz = xp.multiply(V_tan_exp_brxyz, image_brxyz)
                    image_brxyz = self.xp.fft.ifft(self.xp.fft.ifftshift(image_brxyz, axes=per[0][axis]), axis=per[0][axis])        

        return image_brxyz


class Translation_Op(Linop):
    """
    Sinc-interpolated 3D rigid transforms, operating in k-space.
    
    img : 3d-array 
        image, that will be tranformed around the axes in the center of the plane
        give 4x padded image if want to compare to FLIRT rotations
    traj : ndarray 
        of size (N_mot_samples, 7)
        rigid motion trajectory [time, dx, dy, dz, theta_x, theta_y, theta_z]
        motion will be interpolated at the sequence sampling points if interp_motion is True
    Nkm : int
        size of the image
        
    Return 
        image_tr : 3d-array
        Warped image
    """

    def __init__(self, ishape, motion_mp, pad=True, device=None):
        
        warnings.warn('Use Rigid_Motion operator please')
        if not pad or device is not None:
            warnings.warn('pad and device args depreciated, please remove')
        if ishape is None or motion_mp is None:
            raise ValueError('Please provide motion and image shape in Rotation._init_')
            
        assert (ishape[-1] == ishape[-2] == ishape[-3]), 'Input image should be cubic'
            
        self.device = sp.get_device(motion_mp)
        self.xp = self.device.xp
        
        self.ishape = ishape
        self.oshape = tuple([motion_mp.shape[0]] + list(ishape))
        self.mot_traj = motion_mp
        with self.device:
            _, kGrid, _ = generate_grids(ishape[-1], self.xp)
        
        # prep the phase ramp matrices
        self.do_transl = False
        
        shifts = self.mot_traj[:, :3]
        nm = shifts.shape[0]
        
        with self.device:
            if (shifts.any()):
                self.do_transl = True

            if self.do_transl:
                kxx, kyy, kzz = self.xp.meshgrid(kGrid, kGrid, kGrid, indexing='ij')
                dim_k = kGrid.shape[0]
                # dx is along y in Python, dy is along x and with the opposite sign
                self.U_mat = self.xp.exp( - 1j * (self.xp.outer(-shifts[:, 0], kxx) + self.xp.outer(shifts[:, 1], kyy) + \
                               self.xp.outer(shifts[:, 2], kzz)).reshape(nm, dim_k, dim_k, dim_k) )  

        super().__init__(self.oshape, self.ishape)
    
    def _reset_motion_params(self, motion_params):
        """
        motion_params - vector of shape(6,) - translations and rotations
            rotation angles are in [rads]
        """
        raise AttributeError('Class was changed to init with motion')
        
    def _apply(self, input):
        if sp.get_device(input) != self.device:
            raise ResourceWarning(f'Input is on {sp.get_device(input)}, while the current is {self.device}')
            input = sp.to_device(input, self.device) 
            
        with self.device:
            if input.ndim == 3:
                image_brxyz = xp.expand_dims(input, axis=(0, 1))
            elif input.ndim >= 4:
                # bxyz
                image_brxyz = xp.expand_dims(input, axis=-4)

            if self.do_transl:
                U_mat_brxyz = self.U_mat[None,...]
                axes_ = (-1, -2, -3) 
                image_brxyz = self.xp.fft.ifftn( self.xp.fft.ifftshift( U_mat_brxyz * \
                                        self.xp.fft.fftshift(self.xp.fft.fftn\
                                        ( image_brxyz, axes=axes_), \
                                        axes=axes_ ), axes=axes_ ), axes=axes_ )   
        return image_brxyz

class Rigid_Motion(Linop):
    """
    Sinc-interpolated 3D rigid transforms, operating in k-space.
    
    img : 4d-array and 3d-array
        dimensions cxyz or xyz - coil and spatial
        image, that will be tranformed around the axes in the center of the plane
        give 4x padded image if want to compare to FLIRT rotations
    traj : ndarray 
        of size (N_mot_samples, 7)
        rigid motion trajectory [time, dx, dy, dz, theta_x, theta_y, theta_z]
        motion will be interpolated at the sequence sampling points if interp_motion is True
    Nkm : int
        size of the image
        
    Return 
        image_tr : 3d-array
        Warped image
    """

    def __init__(self, ishape=None, motion_mp=None, pad=True, device=None):
        """
        Input image needs to be 2D or 3D cube with dimension N. 
        """
        if not pad or device is not None:
            warnings.warn('pad and device args depreciated, please remove')
        if ishape is None or motion_mp is None:
            raise ValueError('Please provide motion and image shape in Rotation._init_')
            
        assert (ishape[-1] == ishape[-2] == ishape[-3]), 'Input image should be cubic'
            
        self.device = sp.get_device(motion_mp)
        self.xp = self.device.xp
        
        self.ishape = ishape
        self.oshape = tuple([motion_mp.shape[0]] + list(ishape))
        self.mot_traj = motion_mp
        with self.device:
            _, kGrid, rkGrid = generate_grids(ishape[-1], self.xp)
                
        # prep shear matrices 
        self.do_rot = False
        self.do_transl = False
        
        with self.device:
            thetas_mp = self.mot_traj[:, -3:] * self.xp.array([-1, 1, 1]).reshape((1, 3))
            shifts = self.mot_traj[:, :3]
            
            if (thetas_mp.any()):
                self.do_rot = True

            if self.do_rot:
                tantheta2_mp = 1j * self.xp.tan(thetas_mp / 2)
                sintheta_mp = -1j * self.xp.sin(thetas_mp)
                
                # 1. Rotations
                V_mat_tan_mrk = [] # of size 3 - for each rotation axis,
                                   # each element has size (rkGrid_size)
                V_mat_sin_mrk = []
                for ax_ in range(3):
                    exp_mrk = self.xp.exp((tantheta2_mp[:, [2-ax_], None] * rkGrid[None, ...])) 
                    V_mat_tan_mrk.append(exp_mrk)

                    exp_mrk = self.xp.exp((sintheta_mp[:, [2-ax_], None] * rkGrid[None, ...]))
                    V_mat_sin_mrk.append(exp_mrk)  
                self.V_mat_tan_mrk = V_mat_tan_mrk
                self.V_mat_sin_mrk = V_mat_sin_mrk
                
            if (shifts.any()):
                self.do_transl = True

            if self.do_transl:
                nm = self.mot_traj.shape[0]
                kxx, kyy, kzz = self.xp.meshgrid(kGrid, kGrid, kGrid, indexing='ij')
                dim_k = kGrid.shape[0]
                # dx is along y in Python, dy is along x and with the opposite sign
                self.U_mat = self.xp.exp( - 1j * (self.xp.outer(-shifts[:, 0], kxx) + self.xp.outer(shifts[:, 1], kyy) + \
                               self.xp.outer(shifts[:, 2], kzz)).reshape(nm, dim_k, dim_k, dim_k) )  

        
        super().__init__(self.oshape, self.ishape)
    
    def _reset_motion_params(self, motion_params):
        """
        motion_params - vector of shape(6,) - translations and rotations
            rotation angles are in [rads]
        """
        raise AttributeError('Class was changed to init with motion')
    

    def _apply(self, input):
        """
        input is brxyz
        """
        if (sp.get_device(input) != self.device):
            raise ResourceWarning(f'Input is on {sp.get_device(input)}, while the current is {self.device}')
            input = sp.to_device(input, self.device)
            
        with self.device:            

            if input.ndim == 3:
                # xyz
                image_brxyz = self.xp.expand_dims(input, axis=(0, 1))
            elif input.ndim >= 4:
                # bxyz
                image_brxyz = self.xp.expand_dims(input, axis=(-4))
                     
            if self.do_rot:
                per = [[-3, -1, -2], [-2, -3, -1]]
                for axis in [0, 1, 2]:
                    newaxis = 2 - axis + 1
                    V_tan_exp_brxyz = self.xp.expand_dims(self.V_mat_tan_mrk[axis], axis=(newaxis))
                    V_sin_exp_brxyz = self.xp.expand_dims(self.V_mat_sin_mrk[axis], axis=(newaxis))

                    image_brxyz = self.xp.fft.fftshift(self.xp.fft.fft(image_brxyz, axis=per[0][axis]), axes=per[0][axis])
                    image_brxyz =  self.xp.multiply(V_tan_exp_brxyz, image_brxyz) 
                    image_brxyz = self.xp.fft.ifft(self.xp.fft.ifftshift(image_brxyz, axes=per[0][axis]), axis=per[0][axis])

                    image_brxyz = self.xp.fft.fftshift(self.xp.fft.fft(image_brxyz, axis=per[1][axis]), axes=per[1][axis])
                    image_brxyz = self.xp.multiply(V_sin_exp_brxyz, image_brxyz)
                    image_brxyz = self.xp.fft.ifft(self.xp.fft.ifftshift(image_brxyz, axes=per[1][axis]), axis=per[1][axis])

                    image_brxyz = self.xp.fft.fftshift(self.xp.fft.fft(image_brxyz, axis=per[0][axis]), axes=per[0][axis])
                    image_brxyz = self.xp.multiply(V_tan_exp_brxyz, image_brxyz)
                    image_brxyz = self.xp.fft.ifft(self.xp.fft.ifftshift(image_brxyz, axes=per[0][axis]), axis=per[0][axis])        

            if self.do_transl:
                U_mat_brxyz = self.U_mat
                axes_ = (-1, -2, -3) 
                image_brxyz = self.xp.fft.ifftn( self.xp.fft.ifftshift( U_mat_brxyz * \
                                        self.xp.fft.fftshift(self.xp.fft.fftn\
                                        ( image_brxyz, axes=axes_), \
                                        axes=axes_ ), axes=axes_ ), axes=axes_ )  
        return image_brxyz

