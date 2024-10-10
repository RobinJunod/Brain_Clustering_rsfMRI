#%%
import sys
import numpy as np
# import matplotlib.pyplot as plt
import nibabel as nib


def extract_4Ddata_from_nii(nii_file):
    """
    Extract data from nii file
    """
    try:
        print('Loading roi from: ' + nii_file)
        Img = nib.load(nii_file)
        data = Img.get_fdata()
    except:
        sys.exit('Cannot open ' + nii_file | '\nExiting.')
    if len(data.shape) == 4:
        print(nii_file + ' is a 4D image\nExiting.')
    else:
        sys.exit('Data did not loaded successfully')
    return data

def extract_3Ddata_from_nii(nii_file):
    """
    Extract data from nii file
    """
    try:
        print('Loading roi from: ' + nii_file)
        roiImg = nib.load(nii_file)
        data = roiImg.get_fdata()
    except:
        sys.exit('Cannot open ' + nii_file | '\nExiting.')
    if len(data.shape) == 3:
        print(nii_file + ' is a 3D image\nExiting.')
    else:
        sys.exit('Data did not loaded successfully')
    return data


def pca(X):
    from scipy.linalg import svd
    # Center X by subtracting off column means
    X -= np.mean(X,0)
    # The principal components are the eigenvectors of S = X'*X./(n-1), but computed using SVD
    [U,sigma,V] = svd(X,full_matrices=False)
    # Project X onto the principal component axes
    Y = U*sigma
    # Convert the singular values to eigenvalues 
    sigma /= np.sqrt(X.shape[0]-1)
    evals = np.square(sigma)
    
    return V, Y, evals

def corr(X,Y):
    Y = Y.T
    X = X.T
    R = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(0,R.shape[1]):
        y = Y[i,:]
        Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
        ym = np.mean(y)
        r_num = np.sum((X-Xm)*(y-ym),axis=1)
        r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
        R[:,i] = r_num / r_den
    return R

def eta2(X):
    
    S = np.zeros((X.shape[0],X.shape[0]))
    for i in range(0,X.shape[0]):
        for j in range(i,X.shape[0]):
            mi = np.mean([X[i,:],X[j,:]],0) 
            mm = np.mean(mi)
            ssw = np.sum(np.square(X[i,:]-mi) + np.square(X[j,:]-mi))
            sst = np.sum(np.square(X[i,:]-mm) + np.square(X[j,:]-mm))
            S[i,j] = 1-ssw/sst
    
    S += S.T 
    S -= np.eye(S.shape[0])
    
    return S
