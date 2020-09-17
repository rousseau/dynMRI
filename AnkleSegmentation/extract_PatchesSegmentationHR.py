#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:07:42 2020

@author: p20coupe
"""
from tqdm import tqdm
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state
import numpy as np

def array_normalization(X,M=None,norm=0):
  """Normalization for image regression
    Inputs : 
      X : array of data
      M : array of mask used to compute normalization parameters
    Outputs :
      X : normalized data
  """
  if M is None:
    M = np.ones(X.shape)
    
  #normalization using the ROI defined by the mask
        
  if norm == 0:
    #Zero-centered
    X = (X - np.mean(X[M==1])) / np.std(X[M==1])
  else:  
    #[0,1] normalization
    maxX = np.max(X[M==1])
    minX = np.min(X[M==1])
    X = (X - minX)/(maxX-minX)

  return X


def get_hcp_2dpatches_SegmentationHR(extract_pourcent,patch_size = 128, n_patches = 1000, data = None):
  
    (T1s,T2s,T3s,T4s,masks) = data
	
    n_images = len(T2s)
    T1_patches = None
    T2_patches = None
    T3_patches= None
    T4_patches = None

    T1 = []
    T2 = []
    T3 = []
    T4 = []

    mask_extract = extract_pourcent
    patch_shape = (patch_size,patch_size)
    random_state = None

    for i in tqdm(range(n_images)):
		
        #Normalize data using mask
        T2_norm = array_normalization(X=T2s[i],M=masks[i],norm=0)
        mask = masks[i]
        T1_norm = T1s[i]
        T3_norm= T3s[i]
        T4_norm= T4s[i]
		

        for j in range(T1_norm.shape[2]): #Loop over the slices
            pT1 = extract_patches(T1_norm[:,:,j], patch_shape, extraction_step = 1)
            pT2 = extract_patches(T2_norm[:,:,j], patch_shape, extraction_step = 1)
            pT3 = extract_patches(T3_norm[:,:,j], patch_shape, extraction_step = 1)
            pT4 = extract_patches(T4_norm[:,:,j], patch_shape, extraction_step = 1)

            pmask = extract_patches(mask[:,:,j], patch_shape, extraction_step = 1)
            rng = check_random_state(random_state)
            i_s = rng.randint(T1_norm.shape[0] - patch_shape[0] + 1, size = n_patches)
            j_s = rng.randint(T1_norm.shape[1] - patch_shape[1] + 1, size = n_patches)
            pT1 = pT1[i_s, j_s]
            pT2 = pT2[i_s, j_s]
            pT3 = pT3[i_s, j_s]
            pT4 = pT4[i_s, j_s]

            pmask = pmask[i_s, j_s]

            #Channel last
            pT1 = pT1.reshape(-1, patch_shape[0], patch_shape[1])
            pT2 = pT2.reshape(-1, patch_shape[0], patch_shape[1])

            pT3 = pT3.reshape(-1, patch_shape[0], patch_shape[1])
            pT4 = pT4.reshape(-1, patch_shape[0], patch_shape[1])

            pmask = pmask.reshape(-1, patch_shape[0], patch_shape[1])
            pmask = pmask.reshape(pmask.shape[0],-1)

            #Remove empty patches (<65% of mask)
            pmT1 = pT1[ np.mean(pmask,axis=1)>=mask_extract ]
            pmT2 = pT2[ np.mean(pmask,axis=1)>=mask_extract ]
            pmT3 = pT3[ np.mean(pmask,axis=1)>=mask_extract ]
            pmT4 = pT4[ np.mean(pmask,axis=1)>=mask_extract ]
			

            T1.append(pmT1)
            T2.append(pmT2)
            T3.append(pmT3)
            T4.append(pmT4)

    T1_patches = np.concatenate(T1,axis=0)
    T2_patches = np.concatenate(T2,axis=0) 
    T3_patches = np.concatenate(T3,axis=0)
    T4_patches = np.concatenate(T4,axis=0)        

    return (T1_patches,T2_patches,T3_patches,T4_patches)
		
