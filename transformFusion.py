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
from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.morphology import binary_erosion

import time

import multiprocessing



def Text_file_to_matrix(filename):

   T = np.loadtxt(str(filename), dtype='f')

   return np.mat(T)


def Matrix_to_text_file(matrix, text_filename):

    np.savetxt(text_filename, matrix, delimiter='  ')


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
compute the  associated weighting function to a binary mask (a region in the reference image)
Parameters
----------
component : array of data (binary mask)
Returns
-------
output : array
  N_dimensional array containing the weighting function value for each voxel
"""

def component_weighting_function(data):

    data[:,:,:] = np.max(data)-data[:,:,:]  #binary inversion
    distance_function = np.zeros(data.shape)
    weighting_function = np.zeros(data.shape)
    distance_function = ndimage.distance_transform_edt(data)
    weighting_function[:,:,:] = 1/(1+0.5*distance_function[:,:,:])

    return(weighting_function)


"""
transform a point from image 1 to image 2 using a flirt transform matrix
Parameters
----------
x,y,z : the point voxel coordinates in image 1
input_header : image1 header
reference_header : image2 header
transform : ndarray
  flirt transformation, as a 2D array (matrix)
Returns
-------
output : array
  An array 1*3 containing the output point voxel coordinates in image 2
"""



def warp_point_using_flirt_transform(x,y,z,input_header, reference_header, transform):
# flip [x,y,z] if necessary (based on the sign of the sform or qform determinant)
	if np.sign(det(input_header.get_sform()))==1:
		x = input_header.get_data_shape()[0]-1-x
		y = input_header.get_data_shape()[1]-1-y
		z = input_header.get_data_shape()[2]-1-z

#scale the values by multiplying by the corresponding voxel sizes (in mm)
	point=np.ones(4)
	point[0] = x*input_header.get_zooms()[0]
	point[1] = y*input_header.get_zooms()[1]
	point[2] = z*input_header.get_zooms()[2]
# apply the FLIRT matrix to map to the reference space
	point = np.dot(transform, point)

#divide by the corresponding voxel sizes (in mm, of the reference image this time)
	point[0] = point[0]/reference_header.get_zooms()[0]
	point[1] = point[1]/reference_header.get_zooms()[1]
	point[2] = point[2]/reference_header.get_zooms()[2]

#flip the [x,y,z] coordinates (based on the sign of the sform or qform determinant, of the reference image this time)
	if np.sign(det(reference_header.get_sform()))==1:
		point[0] = reference_header.get_data_shape()[0]-1-point[0]
		point[1] = reference_header.get_data_shape()[1]-1-point[1]
		point[2] = reference_header.get_data_shape()[2]-1-point[2]

	return np.transpose(np.absolute(np.delete(point, 3, 0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--floating', help='floating input image', type=str, required = True)

    #parser.add_argument('-roi', '--roi', help='region of interest in the floating HR image', type=str, required = True)

    #parser.add_argument('-ref', '--reference', help='reference image', type=str, required = True)
    parser.add_argument('-refweight', '--component', help='', type=str, required = True,action='append')
    parser.add_argument('-t', '--transform', help='', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)
    parser.add_argument('-warped_image', '--outputimage', help='Output image name', type=str, required = True)
    #parser.add_argument('-erode', '--eroded', help='binary_erosion kernel size', type=int, required = False)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    normalized_weighting_function_path = args.output+'/normalized_weighting_function/'
    if not os.path.exists(normalized_weighting_function_path):
        os.makedirs(normalized_weighting_function_path)


######################compute the normalized weighting function of each component #########################
    nii = nib.load(args.component[0])
    im_gnd = nifti_to_array(args.component[0])
    #ROI = nifti_to_array(args.roi)

    floating_data = nifti_to_array(args.floating)
    dim0, dim1, dim2 = nifti_image_shape(args.floating)
    sum_of_weighting_functions = np.zeros((im_gnd.shape))
    Normalized_weighting_function = np.zeros((im_gnd.shape))


    for i in range(0, len(args.component)):

        sum_of_weighting_functions = np.add(sum_of_weighting_functions, component_weighting_function(nifti_to_array(args.component[i])))

    for i in range(0, len(args.component)):

        Normalized_weighting_function[:,:,:] = np.divide(component_weighting_function(nifti_to_array(args.component[i])), sum_of_weighting_functions)

        k = nib.Nifti1Image(Normalized_weighting_function, nii.affine)
        save_path = normalized_weighting_function_path+'Normalized_weighting_function_component'+str(i)+'.nii.gz'
        nib.save(k, save_path)


###############################################################################################################

###### set of computed normalized weighting functions #######
    Normalized_weighting_functionSet = glob.glob(normalized_weighting_function_path+'*.nii.gz')
    Normalized_weighting_functionSet.sort()

##### create an array of matrices final_transform(x,y,z)= -∑i  w_norm(i)[x,y,z]*log(T(i)) ########


    dim0, dim1, dim2 = nifti_image_shape(args.floating)

    #final_transform = np.zeros((dim0, dim1, dim2, 4, 4), dtype=complex)
    final_transform = np.zeros((dim0, dim1, dim2, 4, 4))



    for i in range(0, len(args.transform)):

        #local_transform[:,:,:] = la.logm(Text_file_to_matrix(args.transform[i]))
        #normalized_weight = nifti_to_array(Normalized_weighting_functionSet[i])
        #final_transform -= local_transform * normalized_weight[:,:,:,newaxis,newaxis]
        final_transform -= np.multiply(la.logm(Text_file_to_matrix(args.transform[i])).real , nifti_to_array(Normalized_weighting_functionSet[i])[:,:,:,newaxis,newaxis])

######## compute the warped image #################################

    header_input = get_header_from_nifti_file(args.floating)


    def warp_point_log_demons(point):

        return(warp_point_using_flirt_transform(point[0] , point[1] , point[2] , header_input , header_input , la.expm(final_transform[point[0],point[1],point[2]]).real))


    print("warped image computing, please wait ...")

#Generate a list of tuples where each tuple is a combination of parameters: Compute the coordinates in the input image
#The list will contain all possible combinations of parameters.

    paramlist = list(itertools.product(range(dim0),range(dim1),range(dim2)))

#Generate processes equal to the number of CPU minus two

    n_CPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_CPU-2)

#Distribute the parameter sets evenly across the cores

    res  = pool.map(warp_point_log_demons, paramlist)

    #pool.close()
    #pool.join()

    print("warped image is successfully computed...")

    coords = np.transpose(np.asarray(res))

    coords = np.resize(coords, (3, dim0, dim1, dim2))


#create index for the reference space

    i = np.arange(0,dim0)
    j = np.arange(0,dim1)
    k = np.arange(0,dim2)
    iv,jv,kv = np.meshgrid(i,j,k,indexing='ij')

    iv = np.reshape(iv,(-1))
    jv = np.reshape(jv,(-1))
    kv = np.reshape(kv,(-1))

#reshape the warped coordinates

    pointset = np.zeros((3,iv.shape[0]))
    pointset[0,:] = iv
    pointset[1,:] = jv
    pointset[2,:] = kv

    coords = np.reshape(coords, pointset.shape)
    val = np.zeros(iv.shape)

#### Interpolation:  mapping output data into the reference image space####

    map_coordinates(nifti_to_array(args.floating),[coords[0,:],coords[1,:],coords[2,:]],output=val,order=1)

    output_data = np.reshape(val,nifti_image_shape(args.floating))


#######writing and saving warped image ###

    i = nib.Nifti1Image(output_data, nii.affine)
    save_path = args.output + args.outputimage
    nib.save(i, save_path)
