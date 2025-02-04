import numpy as np
import pickle

def D(n_navi, order=1):
    D = np.zeros((n_navi, n_navi))
    
    if order == 2:
        for r_ii in range(n_navi):
            D[r_ii, r_ii] = -2.
            if (r_ii != (n_navi - 1)):
                D[r_ii, r_ii+1] = 1.
            if (r_ii != 0 ):
                D[r_ii,r_ii-1] = 1.
                
    elif order == 1:
        for r_ii in range(n_navi):
            D[r_ii, r_ii] = -1.
            if (r_ii != (n_navi - 1)):
                D[r_ii, r_ii+1] = 1.
    return D

def A_mat(mu, mu_GT=0, ng=0, n_navi=160, order=1, Nt_navi=10):
    I = np.eye(n_navi)
    A = np.vstack((I, np.sqrt(mu) * D(n_navi, order)))
    if mu_GT > 0:
        C = np.zeros((ng, n_navi))
        for gr_ii in range(ng):
            C[gr_ii, Nt_navi - 1 + Nt_navi * gr_ii] = 1.
        A = np.vstack((A, np.sqrt(mu_GT) * C))
    return A

def fit_smooth(res_matching, mu_L2_deriv=1.5, mu_L2_GT=0.02, through_GT=True, Nt_navi=1, motion_to_fit='motion_Est', \
               debug=False, save=False, dict_name='dict_name_to_save', order=1):
        """
        The code for smoothening the motion trajectory in a way, that it passes through the provided GT values.
        
        Args:
            res_macthing: dict
                contains motion_Est, motion_GT
                results can be inserted in this dict if save == True
                
            mu_L2_deriv : float
                controls smoothness, lamda for L2 of trajectory derivative
            mu_L2_GT : float
                controls enforcing the GT values
            through_GT : bool
                if need to consider GT values at all
            Nt_navi : int
                number of navigators in one MRF group
            motion_to_fit: str
                can be motion_Est or motion_GT
            order : int
                will be extended later to choose between 1st and 2nd derivative for smoothening
            
            
        """
        motion_rp_fit = np.zeros_like(res_matching['motion_Est'])
        n_navi = motion_rp_fit.shape[0] + Nt_navi # to avoid edge effects of regularization
        ng = n_navi // Nt_navi
        
        for par in np.arange(6):
            theta_r = res_matching[motion_to_fit][:, par]
            theta_r = np.hstack((theta_r, theta_r[-Nt_navi:]))
            theta_GT = res_matching['motion_GT'][Nt_navi-1::Nt_navi, par]  
            theta_GT = np.hstack((theta_GT, theta_GT[-1]))
            
            y = np.hstack((theta_r, np.zeros((n_navi, ))))
            if mu_L2_GT > 0:
                y = np.hstack((y, theta_GT))

            A = A_mat(mu_L2_deriv, mu_L2_GT, ng=ng, n_navi=n_navi, order=order)
#             print(f'A and y {A.shape, y.shape}')
            
            C = np.zeros((ng, n_navi))
            if through_GT:
                for gr_ii in range(ng):
                    C[gr_ii, Nt_navi - 1 + Nt_navi * gr_ii] = 1.  
                B = np.block([ [A.T @ A, C.T],  [C, np.zeros((ng, ng))] ])
                b = np.hstack((A.T @ y, theta_GT))
                
            else:
                B = A.T @ A
                b = A.T @ y
        
#             print(B.shape, b.shape)
            g = (np.linalg.pinv(B) @ b)[:n_navi-Nt_navi]    
            motion_rp_fit[:, par] = np.copy(g)
            
            if debug:
                plt.figure()
                plt.plot(theta_r, '-o')
                plt.plot(g, '-')
                plt.plot(9 + np.arange(0, 160, 10), theta_GT, 'x')
                
            if save:
                if motion_to_fit == 'motion_Est':
                    if through_GT:
                        res_matching['motion_smooth_GT'] = np.copy(motion_rp_fit)
                    else:
                        res_matching['motion_smooth_noGT'] = np.copy(motion_rp_fit)
                elif motion_to_fit == 'motion_GT':
                    res_matching['motion_GT_smooth'] = np.copy(motion_rp_fit)
                with open(dict_name, 'wb') as f:
                    pickle.dump(res_matching, f)
        return res_matching
    
