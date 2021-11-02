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

from __future__ import division
from scipy import ndimage
import nibabel as nib
import numpy as np
import argparse
import os
import scipy.linalg as la
import glob
from numpy.linalg import det
from numpy import newaxis
import itertools
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_erosion
import multiprocessing


from numpy import *



def Text_file_to_matrix(filename):

   T = np.loadtxt(str(filename), dtype='f')

   return np.mat(T)


def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())


def nifti_image_shape(filename):

    nii = nib.load(filename)
    data = nii.get_data()

    return (data.shape)


def get_header_from_nifti_file(filename):

    nii = nib.load(filename)

    return nii.header

"""
Define the sigmoid function,  which smooths the  slope of the  weight map near the wire.
Parameters
----------
x : N dimensional array
Returns
-------
output : array
  N_dimensional array containing sigmoid function result
"""

def sigmoid(x):

  return 1 / (1 + np.exp(np.negative(x)))



def distance_to_mask(mask):

    d = np.subtract(np.max(mask), mask)

    return ndimage.distance_transform_edt(d)

"""
compute the  associated weighting function to a binary mask (a region in the reference image)
Parameters
----------
component : array of data (binary mask)
Returns
-------
output : array
  N_dimensional array containing the weighting function value for each voxel according to the entered mask, convolved with a Gaussian kernel
  with a standard deviation set to 2 voxels inorder to take into account the partial volume effect due to anisotropy of the image resolution
"""

def component_weighting_function(data):

    #return 2/(1+np.exp(0.4*distance_to_mask(data)))
    return 1/(1+0.5*distance_to_mask(data)**2)


"""
The scipy.linalg.logm method in the scipy library of Python2.7 calculates matrix exponentials via the Padé approximation.
However, using eigendecomposition to calculate the logarithm of a 4*4 matrix is more accurate and is faster by approximately a factor of 2.
"""

def matrix_logarithm(matrix):

    d, Y = np.linalg.eig(matrix)
    Yinv = np.linalg.inv(Y)
    D = np.diag(np.log(d))
    Y = np.asmatrix(Y)
    D = np.asmatrix(D)
    Yinv = np.asmatrix(Yinv)

    return np.array(np.dot(Y,np.dot(D,Yinv))).reshape(4,4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--floating', help='floating input image', type=str, required = True)
    parser.add_argument('-refweight', '--component', help='bone masks in the target image', type=str, required = True,action='append')
    parser.add_argument('-t', '--transform', help='bone transforms from source to target image', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)
    parser.add_argument('-warped_image', '--outputimage', help='Output image name', type=str, default='Warped_image.nii.gz')
    parser.add_argument('-def_field', '--deformation_field', help='Deformation field image name', type=str, default='Deformation_field.nii.gz')
    parser.add_argument('-tempinterp', '--temporal_interpolation', help='Temporal interpolation of the estimated deformation field. Example:\
    if this argument value is set to 2, the algorithm will return the deformation field half way between the source and the target images'\
    , type=int, default=1)
    parser.add_argument('-ordinterp', '--interp_ord', help='(optional): The order of the spline interpolation when mapping input\
    image intensities to the reference space, default is 3. The order has to be in the range 0-5.', type=int, default=3)


    args = parser.parse_args()

    t = 1/args.temporal_interpolation


    if not os.path.exists(args.output):
        os.makedirs(args.output)

    normalized_weighting_function_path = args.output+'/normalized_weighting_function/'
    if not os.path.exists(normalized_weighting_function_path):
        os.makedirs(normalized_weighting_function_path)

######################compute the normalized weighting function of each component #########################
    nii = nib.load(args.component[0])
    data_shape = nifti_image_shape(args.component[0])
    dim0, dim1, dim2 = nifti_image_shape(args.floating)

######################## automatically identify border voxels #############################################

    #borders = np.ones((dim0, dim1, dim2))
    #border_width = 15
    #borders[border_width:dim0-border_width,border_width:dim1-border_width,:] = 0

######################## Compute and save normalized weighting functions ##################################

    #sum_of_weighting_functions = component_weighting_function(borders)

    sum_of_weighting_functions = np.zeros((data_shape))

    Normalized_weighting_function = np.zeros((data_shape))

    for i in range (len(args.component)):

        sum_of_weighting_functions += component_weighting_function(nifti_to_array(args.component[i]))


    #np.divide(component_weighting_function(borders), sum_of_weighting_functions, Normalized_weighting_function)
    #k = nib.Nifti1Image(Normalized_weighting_function, nii.affine)
    #save_path = normalized_weighting_function_path+'Normalized_weighting_function_component0.nii.gz'
    #nib.save(k, save_path)

    for i in range (len(args.component)):

        np.divide(component_weighting_function(nifti_to_array(args.component[i])), sum_of_weighting_functions,\
        Normalized_weighting_function)

        k = nib.Nifti1Image(Normalized_weighting_function, nii.affine)
        save_path = normalized_weighting_function_path+'Normalized_weighting_function_component'+str(i)+'.nii.gz'
        nib.save(k, save_path)

###############################################################################################################
    del sum_of_weighting_functions
    del Normalized_weighting_function
    #del borders

###### set of computed normalized weighting functions #######
    Normalized_weighting_functionSet = glob.glob(normalized_weighting_function_path+'*.nii.gz')
    Normalized_weighting_functionSet.sort()

##### create an array of matrices: T(x,y,z)= ∑i  w_norm(i)[x,y,z]*log(T(i)) ########
    T = np.zeros((dim0, dim1, dim2, 4, 4))

    for i in range (len(args.transform)):
        #np.subtract(T, np.multiply(la.logm(Text_file_to_matrix(args.transform[i])).real , nifti_to_array\
        #(Normalized_weighting_functionSet[i+1])[:,:,:,newaxis,newaxis]), T)
        np.add(T, np.multiply(la.logm(Text_file_to_matrix(args.transform[i])).real ,t*nifti_to_array\
        (Normalized_weighting_functionSet[i])[:,:,:,newaxis,newaxis]), T)

    print("principal matrix logarithm of each bone transformation was successfully computed")

######## compute the exponential of each matrix in the final_log_transform array of matrices using Eigen decomposition   #####
##############################  final_exp_transform(T(x,y,z))= exp(-∑i  w_norm(i)[x,y,z]*log(T(i))) ##########################

    d, Y = np.linalg.eig(T)  #returns an array of vectors with the eigenvalues (d[dim0,dim1,dim2,4]) and an array
                             #of matrices (Y[dim0,dim1,dim2,(4,4)]) with corresponding eigenvectors
    print("eigenvalues and eigen vectors were successfully computed")

    Yi = np.linalg.inv(Y)
    print("eigenvectors were successfully inverted")

    d = np.exp(d)  # exp(T) = Y*exp(d)*inv(Y).  exp (d) is much more easy to calculate than exp(T)
    #since, for a diagonal matrix, we just need to exponentiate the diagonal elements.
    print("exponentiate the diagonal elements (complex eigenvalues): done")

    #first row
    T[...,0,0] = Y[...,0,0]*Yi[...,0,0]*d[...,0] + Y[...,0,1]*Yi[...,1,0]*d[...,1] + Y[...,0,2]*Yi[...,2,0]*d[...,2]
    T[...,0,1] = Y[...,0,0]*Yi[...,0,1]*d[...,0] + Y[...,0,1]*Yi[...,1,1]*d[...,1] + Y[...,0,2]*Yi[...,2,1]*d[...,2]
    T[...,0,2] = Y[...,0,0]*Yi[...,0,2]*d[...,0] + Y[...,0,1]*Yi[...,1,2]*d[...,1] + Y[...,0,2]*Yi[...,2,2]*d[...,2]
    T[...,0,3] = Y[...,0,0]*Yi[...,0,3]*d[...,0] + Y[...,0,1]*Yi[...,1,3]*d[...,1] + Y[...,0,2]*Yi[...,2,3]*d[...,2] \
    + Y[...,0,3]*Yi[...,3,3]
    #second row
    T[...,1,0] = Y[...,1,0]*Yi[...,0,0]*d[...,0] + Y[...,1,1]*Yi[...,1,0]*d[...,1] + Y[...,1,2]*Yi[...,2,0]*d[...,2]
    T[...,1,1] = Y[...,1,0]*Yi[...,0,1]*d[...,0] + Y[...,1,1]*Yi[...,1,1]*d[...,1] + Y[...,1,2]*Yi[...,2,1]*d[...,2]
    T[...,1,2] = Y[...,1,0]*Yi[...,0,2]*d[...,0] + Y[...,1,1]*Yi[...,1,2]*d[...,1] + Y[...,1,2]*Yi[...,2,2]*d[...,2]
    T[...,1,3] = Y[...,1,0]*Yi[...,0,3]*d[...,0] + Y[...,1,1]*Yi[...,1,3]*d[...,1] + Y[...,1,2]*Yi[...,2,3]*d[...,2] \
    + Y[...,1,3]*Yi[...,3,3]
    #third row
    T[...,2,0] = Y[...,2,0]*Yi[...,0,0]*d[...,0] + Y[...,2,1]*Yi[...,1,0]*d[...,1] + Y[...,2,2]*Yi[...,2,0]*d[...,2]
    T[...,2,1] = Y[...,2,0]*Yi[...,0,1]*d[...,0] + Y[...,2,1]*Yi[...,1,1]*d[...,1] + Y[...,2,2]*Yi[...,2,1]*d[...,2]
    T[...,2,2] = Y[...,2,0]*Yi[...,0,2]*d[...,0] + Y[...,2,1]*Yi[...,1,2]*d[...,1] + Y[...,2,2]*Yi[...,2,2]*d[...,2]
    T[...,2,3] = Y[...,2,0]*Yi[...,0,3]*d[...,0] + Y[...,2,1]*Yi[...,1,3]*d[...,1] + Y[...,2,2]*Yi[...,2,3]*d[...,2] \
    + Y[...,2,3]*Yi[...,3,3]
    #fourth row
    #T[...,3,3]= 1 #in homogeneous coordinates

    ### Remove Y, Yi, and d from the computer RAM
    del Y
    del Yi
    del d

    print("final exponential mapping is successfully computed")

######## compute the warped image #################################

    in_header = get_header_from_nifti_file(args.floating)
    ref_header = get_header_from_nifti_file(args.floating)

# Compute coordinates in the input image
    coords = np.zeros((3,dim0, dim1, dim2), dtype='float32')
    coords[0,...] = np.arange(dim0)[:,newaxis,newaxis]
    coords[1,...] = np.arange(dim1)[newaxis,:,newaxis]
    coords[2,...] = np.arange(dim2)[newaxis,newaxis,:]

# Flip [x,y,z] if necessary (based on the sign of the sform or qform determinant)
    if np.sign(det(in_header.get_qform())) == 1:
        coords[0,...] = in_header.get_data_shape()[0]-1-coords[0,...]
        coords[1,...] = in_header.get_data_shape()[1]-1-coords[1,...]
        coords[2,...] = in_header.get_data_shape()[2]-1-coords[2,...]

# Scale the values by multiplying by the corresponding voxel sizes (in mm)
    np.multiply(in_header.get_zooms()[0], coords[0,...], coords[0,...])
    np.multiply(in_header.get_zooms()[1], coords[1,...], coords[1,...])
    np.multiply(in_header.get_zooms()[2], coords[2,...], coords[2,...])

# Apply the FLIRT matrix for each voxel to map to the reference space
# Compute velocity vector fields
    coords_ref = np.zeros((3,dim0, dim1, dim2),dtype='float32')
    coords_ref[0,...] = T[...,0,0]*coords[0,...] + T[...,0,1]*coords[1,...] + T[...,0,2]* coords[2,...] +  T[...,0,3]
    coords_ref[1,...] = T[...,1,0]*coords[0,...] + T[...,1,1]*coords[1,...] + T[...,1,2]* coords[2,...] +  T[...,1,3]
    coords_ref[2,...] = T[...,2,0]*coords[0,...] + T[...,2,1]*coords[1,...] + T[...,2,2]* coords[2,...] +  T[...,2,3]

# Remove final transforms from the computer RAM after computing the vector velocity field
    del T

# Divide by the corresponding voxel sizes (in mm, of the reference image this time)
    np.divide(coords_ref[0,...], ref_header.get_zooms()[0], coords[0,...])
    np.divide(coords_ref[1,...], ref_header.get_zooms()[1], coords[1,...])
    np.divide(coords_ref[2,...], ref_header.get_zooms()[2], coords[2,...])

    del coords_ref

# Flip the [x,y,z] coordinates (based on the sign of the sform or qform determinant, of the reference image this time)
    if np.sign(det(ref_header.get_qform())) == 1:
        coords[0,...] = ref_header.get_data_shape()[0]-1-coords[0,...]
        coords[1,...] = ref_header.get_data_shape()[1]-1-coords[1,...]
        coords[2,...] = ref_header.get_data_shape()[2]-1-coords[2,...]

    print("warped image is successfully computed")

# Compute the deformation field
    def_field = np.concatenate((coords[0,...,newaxis],coords[1,...,newaxis], coords[2,...,newaxis]),axis=3)
    # 4 dimensional volume ... each image in the volume describes the deformation with respect to a specific direction: x, y or z

# Create index for the reference space
    i = np.arange(0,dim0)
    j = np.arange(0,dim1)
    k = np.arange(0,dim2)
    iv,jv,kv = np.meshgrid(i,j,k,indexing='ij')

    iv = np.reshape(iv,(-1))
    jv = np.reshape(jv,(-1))
    kv = np.reshape(kv,(-1))

# Reshape the warped coordinates
    pointset = np.zeros((3,iv.shape[0]))
    pointset[0,:] = iv
    pointset[1,:] = jv
    pointset[2,:] = kv

    coords = np.reshape(coords, pointset.shape)
    val = np.zeros(iv.shape)

#### Interpolation:  mapping output data into the reference image space by spline interpolation of the requested order ####
    map_coordinates(nifti_to_array(args.floating),[coords[0,:],coords[1,:],coords[2,:]],output=val,order=args.interp_ord\
    , mode='nearest')

    del coords
    output_data = np.reshape(val,nifti_image_shape(args.floating))

#######writing and saving warped image ###
    i = nib.Nifti1Image(output_data, nii.affine)
    save_path = args.output + args.outputimage
    nib.save(i, save_path)

    j = nib.Nifti1Image(def_field, nii.affine)
    save_path2 = args.output + args.deformation_field #'4D_def_field.nii.gz'
    nib.save(j, save_path2)
