from tqdm import tqdm
from sklearn.feature_extraction.image import extract_patches
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


def get_hcp_coupes2d(data = None):
  
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

	for i in tqdm(range(n_images)):
		#Normalize data using mask
		T2_norm = array_normalization(X=T2s[i],M=masks[i],norm=0)
		mask = masks[i]
		T1_norm = T1s[i]

		T3_norm= T3s[i]
		T4_norm= T4s[i]
		

		for j in range(T1_norm.shape[2]): #Loop over the slices
			pT1 = extract_patches(T1_norm[:,:,j], T1_norm.shape[0], extraction_step = 1)
			pT2 = extract_patches(T2_norm[:,:,j], T1_norm.shape[0], extraction_step = 1)

			pT3 = extract_patches(T3_norm[:,:,j], T1_norm.shape[0], extraction_step = 1)
			pT4 = extract_patches(T4_norm[:,:,j], T1_norm.shape[0], extraction_step = 1)

			pmask = extract_patches(mask[:,:,j], T1_norm.shape[0], extraction_step = 1)
			
			#Channel last
			pT1 = pT1.reshape(-1, T1_norm.shape[0], T1_norm.shape[1])
			pT2 = pT2.reshape(-1, T1_norm.shape[0], T1_norm.shape[1])

			pT3 = pT3.reshape(-1, T1_norm.shape[0], T1_norm.shape[1])
			pT4 = pT4.reshape(-1, T1_norm.shape[0], T1_norm.shape[1])

			pmask = pmask.reshape(-1, T1_norm.shape[0], T1_norm.shape[1])
			pmask = pmask.reshape(pmask.shape[0],-1)
			

			T1.append(pT1)
			T2.append(pT2)

			T3.append(pT3)
			T4.append(pT4)

		T1_patches = np.concatenate(T1,axis=0)
		T2_patches = np.concatenate(T2,axis=0) 

		T3_patches = np.concatenate(T3,axis=0)
		T4_patches = np.concatenate(T4,axis=0)        

		return (T1_patches,T2_patches,T3_patches,T4_patches)

