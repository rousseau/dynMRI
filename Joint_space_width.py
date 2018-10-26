# -*- coding: utf-8 -*-
"""
  © IMT Atlantique - LATIM-INSERM UMR 1101
  Author(s): Karim Makki (karim.makki@imt-atlantique.fr)

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.
"""

"""
% 3Dimensional Eulerian  framework  for  computing the thickness of tissues

% between two simply connected boundaries

% Inputs: segmentations of the two boundaries and of the tissue. Segmentations must

 have the same dimensions

% Outputs: The harmonic function u, the arclength L0 of the correspondence trajectory
  between each tissue voxel x and the first boundary, the arclength L1 of the correspondence
  trajectory between each tissue voxel x and the second boundary, and the tissue thickness
  W = L0+L1. (JSW)
  Results will be saved as 3D nifti files.
"""


import numpy as np
import nibabel as nib
from numpy import linalg as la
import argparse
import os
from scipy.ndimage.filters import gaussian_filter



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--anatomical', help='anatomical MR volume', type=str, required = True)
    parser.add_argument('-tissue', '--tissue', help='binary mask of the segmented tissue (R)', type=str, required = True)
    parser.add_argument('-boundary', '--boundary', help='binary mask for each boundary',type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='output directory', type=str, required = True)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    r = nib.load(args.tissue).get_fdata()
    S = nib.load(args.boundary[0]).get_fdata()
    S_prime = nib.load(args.boundary[1]).get_fdata()

    R = np.where(r != 0)

    print(len(R[0]))

    nx, ny, nz = S.shape #Number of steps in space(x), (y) and (z)

    r[...,0] = 0
    r[...,nz-1] = 0

	#Initial conditions

    hdr = nib.load(args.anatomical).header # Anatomical image header: this will give the voxel spacing along each axis (dx, dy, dz)
    dx = hdr.get_zooms()[0] #x-voxel spacing
    dy = hdr.get_zooms()[1] #y-voxel spacing
    dz = hdr.get_zooms()[2] #z-voxel spacing

    voxel_volume = dx * dy * dz

    u = np.zeros((nx,ny,nz))  #u(i)
    u_n = np.zeros((nx,ny,nz)) #u(i+1)

    L0 = np.zeros((nx,ny,nz))
    L1 = np.zeros((nx,ny,nz))


    L0_n = np.zeros((nx,ny,nz))
    L1_n = np.zeros((nx,ny,nz))

    Nx = np.zeros((nx,ny,nz))
    Ny = np.zeros((nx,ny,nz))
    Nz = np.zeros((nx,ny,nz))


    thickness = np.zeros((nx,ny,nz))

	#Dirichlet boundary conditions

    u[np.where(S_prime != 0)] =  100

	# Boundary conditions for computing L0 and L1

    L0[:,:,:]= -(dx+dy+dz)/6
    L1[:,:,:]= -(dx+dy+dz)/6

	# Explicit iterative scheme to solve the Laplace equation by the simplest method  (the  Jacobi method).
	# We obtain the harmonic function u(x,y,z) by solving the Laplace equation over R

    n_iter = 200

    for it in range (n_iter):

        u_n = u

        u[R[0],R[1],R[2]]= (u_n[R[0]+1,R[1],R[2]] + u_n[R[0]-1,R[1],R[2]] +  u_n[R[0],R[1]+1,R[2]] \
        + u_n[R[0],R[1]-1,R[2]] + u_n[R[0],R[1],R[2]+1] + u_n[R[0],R[1],R[2]-1]) / 6


	del u_n

    print("Laplacian equation is solved")

	##Compute the normalized tangent vector field of the correspondence trajectories

    N_xx, N_yy, N_zz = np.gradient(u)

    grad_norm = np.sqrt(N_xx**2 + N_yy**2 + N_zz**2)

    grad_norm[np.where(grad_norm==0)] = 1 ## to avoid dividing by zero

	# Normalization
    np.divide(N_xx, grad_norm, Nx)
    np.divide(N_yy, grad_norm, Ny)
    np.divide(N_zz, grad_norm, Nz)

    #gaussian_filter(Nx, sigma=1, output=Nx)
    #gaussian_filter(Ny, sigma=1, output=Ny)
    #gaussian_filter(Nz, sigma=1, output=Nz)

    del grad_norm, N_xx, N_yy, N_zz

    print("The normalized tangent vector field is successfully computed")

    den = np.absolute(Nx)+ np.absolute(Ny) + np.absolute(Nz)

    den[np.where(den==0)] = 1 ## to avoid dividing by zero

	# iteratively compute correspondence trajectory lengths L0 and L1

    for it in range (100):

        L0_n = L0
        L1_n = L1

        L0[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L0_n[(R[0]-np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
        np.absolute(Ny[R[0],R[1],R[2]]) * L0_n[R[0],(R[1]-np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
        + np.absolute(Nz[R[0],R[1],R[2]]) * L0_n[R[0],R[1],(R[2]-np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]

        L1[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L1_n[(R[0]+np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
        np.absolute(Ny[R[0],R[1],R[2]]) * L1_n[R[0],(R[1]+np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
        + np.absolute(Nz[R[0],R[1],R[2]]) * L1_n[R[0],R[1],(R[2]+np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]


	del L0_n, L1_n
	# compute  the thickness of the tissue region inside R

    thickness[R[0],R[1],R[2]] = L0[R[0],R[1],R[2]] + L1[R[0],R[1],R[2]]

    print("Mean thickness inside R:\n")
    print(np.mean(thickness[R[0],R[1],R[2]]))
    print("Maximum thickness inside R:\n")
    print(np.max(thickness[R[0],R[1],R[2]]))
    print("Minimum thickness inside R:\n")
    print(np.min(thickness[R[0],R[1],R[2]]))

    ## To reduce the effects of voxelisation when computing the finite differences, we convolve the result with a 3D Gaussian filter

    #gaussian_filter(thickness, sigma=(0.5, 0.5, 0.5), output=thickness)

    print("Thickness between boundaries is successfully computed")

	# Save results as 3D nifti files

    nii = nib.load(args.boundary[0])

    i = nib.Nifti1Image(u, nii.affine)
    j = nib.Nifti1Image(thickness, nii.affine)
    k = nib.Nifti1Image(L0, nii.affine)
    l = nib.Nifti1Image(L1, nii.affine)


    nib.save(i, args.output + '/harmonic_function.nii.gz')
    nib.save(j, args.output + '/thickness.nii.gz')
    nib.save(k, args.output + '/L0.nii.gz')
    nib.save(l, args.output + '/L1.nii.gz')
