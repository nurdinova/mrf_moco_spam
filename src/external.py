from IPython.display import clear_output
import numpy as np
import subprocess
import cfl
import matplotlib.pyplot as plt
import nibabel as nib

def call_flirt(im, im_ref, pre_saved=False, im_out_name="im_reg", regMat_name="reg", save_out=True):
    """
    Register magnitude of the provided complex images and 
    save the tranformation matrix in 'regMat_name.nat' and the registered image in 'im_out_name' 
    
    Return
        im_reg
    Prints put the resulting matrix
    """
    if not pre_saved:
        affine = np.diag([1,1,1,1]) # res and dt
        array_img_p2 = nib.Nifti1Image(np.abs(im), affine)
        array_img_p3 = nib.Nifti1Image(np.abs(im_ref), affine)

        nib.save(array_img_p2, "img_pose1.nii")
        nib.save(array_img_p3, "img_pose2.nii")
    
    subprocess.run(["/usr/local/fsl/bin/flirt -in img_pose2 -ref img_pose1 \
            -out "+im_out_name+" -omat "+regMat_name+".mat -dof 6 -interp spline"]\
                  , shell=True)
    out_img = nib.load(im_out_name+".nii.gz").get_fdata()
    
    regMat = np.loadtxt(regMat_name+".mat")
    
    output = subprocess.check_output(["avscale --allparams "+regMat_name+".mat"], shell=True)
    # parse the output to get motion params
    for row in output.decode().split('\n'):
        print(row)
        if 'Rotation Angles (x,y,z) [rads]' in row:
            _,value = row.split(' = ')
            angles = value.split(" ")
        if 'Translations (x,y,z) [mm]' in row:
            _, value = row.split(' = ')
            shifts = value.split(" ")
    motion = np.concatenate((np.asarray(shifts[:-1], dtype=float), np.asarray(angles[:-1], dtype=float)))
            
    if not save_out:
        subprocess.run(["rm img_pose1.nii img_pose2.nii "+im_out_name+".nii "+regMat_name+".mat "]\
                  , shell=True)

    clear_output()
    return out_img, motion

def gen_nii(im, voxel_size, file_name):
    affine = np.diag(voxel_size) # res and dt
    array_img_p2 = nib.Nifti1Image((im), affine)
    nib.save(array_img_p2, file_name)

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def get_motion_afni(filename, N_poses):
    """
    output in mm and degs
    """
    output = subprocess.check_output(["cat "+filename+".1D"], shell=True)
    motion_estim_ = np.zeros((N_poses, 6))
    for row_ii, row in enumerate(output.decode().split('\n')):
        value = remove_values_from_list(row.split(' '), '')
        if (len(value) > 6):
            values_float = np.asarray(value[1:], dtype=float)
        elif (len(value) == 6):
            values_float = np.asarray(value, dtype=float)
        else:
            continue
        xyz_rot_ind = np.int32([1,2,0])
        xyz_transl_ind = np.int32([4,5,3])
        angles = values_float[xyz_rot_ind] * [-1, -1, -1]
        shifts = values_float[xyz_transl_ind] * [-1, 1, -1]
        if row_ii >= 1:
            motion_estim_[row_ii,:] = np.concatenate((shifts,angles))
    return motion_estim_

def call_afni(img_xyzp, pre_saved=False, save_out=True):
    """
    Register magnitude of the provided complex images and 
    save the tranforation matrix in 'regMat_name.nat' and the registered image in 'im_out_name' 
    
    Return
        im_reg
    Prints put the resulting matrix
    """
    N_poses = img_xyzp.shape[-1]
    if not pre_saved:
        gen_nii(np.abs(img_xyzp), [1,1,1,1], "img_afni_xyzp.nii")

    subprocess.run(["/home/xcao/abin/3dvolreg -Fourier -maxite 1000 -prefix img_reg_afni_xyzp \
             -base 0 -1Dfile motion_afni.1D img_afni_xyzp.nii"], shell=True)
    
    out_img = nib.load("img_reg_afni_xyzp+tlrc.BRIK").get_fdata()
    motion = get_motion_afni("motion_afni", N_poses)    
    
    if not save_out:
        subprocess.run(["rm img_reg_afni_xyzp* img_afni_xyzp.nii motion_afni "], shell=True)

    return out_img, motion