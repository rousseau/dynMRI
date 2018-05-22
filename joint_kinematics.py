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

import SimpleITK as sitk
import glob
import numpy as np
import os
from numpy.linalg import inv
from numpy.linalg import det
import xlwt
from xlwt import Workbook
import xlrd
import math
import argparse
import nibabel as nib
from scipy.ndimage.interpolation import map_coordinates
from numpy import linalg as LA


"""
applyxfm: Apply a flirt transform matrix to an input image
Parameters
----------
input image: input nifti file
reference image: reference nifti file
transform: ndarray
  flirt transformation, as a 2D array (matrix)
output image: output file full-path
mode : str
Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’, ‘mirror’ or ‘wrap’).

Returns
-------
output :
warped image as nifti file
"""

def applyxfm(input_image, reference_image, output_image, transform, mode):

    ## loading input image
    nii_in = nib.load(input_image)
    ## loading reference image
    nii_ref = nib.load(reference_image)
    input_header = nii_in.header
    reference_header = nii_ref.header
    input_data = nii_in.get_data()
    dim0, dim1, dim2 = input_data.shape
    reference_image = nii_ref.get_data()
    diim0, diim1, diim2 = reference_image.shape
    transform= inv(transform)

## Compute coordinates in the input image

    coords = np.zeros((3,dim0, dim1, dim2))
    coords[0,...] = np.arange(dim0)[:,np.newaxis,np.newaxis]
    coords[1,...] = np.arange(dim1)[np.newaxis,:,np.newaxis]
    coords[2,...] = np.arange(dim2)[np.newaxis,np.newaxis,:]

# flip [x,y,z] if necessary (based on the sign of the sform or qform determinant)

    if np.sign(det(input_header.get_qform())) == 1:
        coords[0,...] = input_header.get_data_shape()[0]-1-coords[0,...]
        coords[1,...] = input_header.get_data_shape()[1]-1-coords[1,...]
        coords[2,...] = input_header.get_data_shape()[2]-1-coords[2,...]

#scale the values by multiplying by the corresponding voxel sizes (in mm)

    np.multiply(input_header.get_zooms()[0], coords[0,...], coords[0,...])
    np.multiply(input_header.get_zooms()[1], coords[1,...], coords[1,...])
    np.multiply(input_header.get_zooms()[2], coords[2,...], coords[2,...])

# apply the FLIRT matrix to map to the reference space

    coords_ref = np.zeros((3,dim0, dim1, dim2))

    coords_ref[0,...] = transform.item((0,0))*coords[0,...] + transform.item((0,1))*coords[1,...] + transform.item((0,2))* coords[2,...] +  transform.item((0,3))
    coords_ref[1,...] = transform.item((1,0))*coords[0,...] + transform.item((1,1))*coords[1,...] + transform.item((1,2))* coords[2,...] +  transform.item((1,3))
    coords_ref[2,...] = transform.item((2,0))*coords[0,...] + transform.item((2,1))*coords[1,...] + transform.item((2,2))* coords[2,...] +  transform.item((2,3))

#divide by the corresponding voxel sizes (in mm, of the reference image this time)

    np.divide(coords_ref[0,...], reference_header.get_zooms()[0], coords[0,...])
    np.divide(coords_ref[1,...], reference_header.get_zooms()[1], coords[1,...])
    np.divide(coords_ref[2,...], reference_header.get_zooms()[2], coords[2,...])

#flip the [x,y,z] coordinates (based on the sign of the sform or qform determinant, of the reference image this time)

    if np.sign(det(reference_header.get_qform())) == 1:
        coords[0,...] = reference_header.get_data_shape()[0]-1-coords[0,...]
        coords[1,...] = reference_header.get_data_shape()[1]-1-coords[1,...]
        coords[2,...] = reference_header.get_data_shape()[2]-1-coords[2,...]


    #create index for the reference space
    i = np.arange(0,diim0)
    j = np.arange(0,diim1)
    k = np.arange(0,diim2)
    iv,jv,kv = np.meshgrid(i,j,k,indexing='ij')

    iv = np.reshape(iv,(-1))
    jv = np.reshape(jv,(-1))
    kv = np.reshape(kv,(-1))

    #reshape the warped coordinates
    pointset = np.zeros((3,iv.shape[0]))
    pointset[0,:] = iv
    pointset[1,:] = jv
    pointset[2,:] = kv

    coords= np.reshape(coords, pointset.shape)
    val = np.zeros(iv.shape)

### Interpolation:  mapping input data into the reference image space

    map_coordinates(input_data,[coords[0,:],coords[1,:],coords[2,:]],output=val,order=1, mode=mode)

    output_data = np.reshape(val,reference_image.shape)

#writing and saving warped image

    s = nib.Nifti1Image(output_data, nii_ref.affine)
    nib.save(s,output_image)

    return 0


def warp_point_using_flirt_transform(point,input_header, reference_header, transform):
# flip [x,y,z] if necessary (based on the sign of the sform or qform determinant)
	if np.sign(det(input_header.get_sform()))==1:
		point[0] = input_header.get_data_shape()[0]-1-point[0]
		point[1] = input_header.get_data_shape()[1]-1-point[1]
		point[2] = input_header.get_data_shape()[2]-1-point[2]
#scale the values by multiplying by the corresponding voxel sizes (in mm)
	p=np.ones((4))
    #point = np.ones(4)
	p[0] = point[0]*input_header.get_zooms()[0]
	p[1] = point[1]*input_header.get_zooms()[1]
	p[2] = point[2]*input_header.get_zooms()[2]
# apply the FLIRT matrix to map to the reference space
	p = np.dot(transform, p[:,np.newaxis])
    #print(point.shape)

#divide by the corresponding voxel sizes (in mm, of the reference image this time)
	p[0, np.newaxis] /= reference_header.get_zooms()[0]
	p[1, np.newaxis] /= reference_header.get_zooms()[1]
	p[2, np.newaxis] /= reference_header.get_zooms()[2]

#flip the [x,y,z] coordinates (based on the sign of the sform or qform determinant, of the reference image this time)
	if np.sign(det(reference_header.get_sform()))==1:
		p[0, np.newaxis] = reference_header.get_data_shape()[0]-1-p[0, np.newaxis]
		p[1, np.newaxis] = reference_header.get_data_shape()[1]-1-p[1, np.newaxis]
		p[2, np.newaxis] = reference_header.get_data_shape()[2]-1-p[2, np.newaxis]

	return np.absolute(np.delete(p, 3, 0))


##Given the  XYZ orthogonal coordinate system (image coordinate system), find a transformation, M, that maps XYZ  to an anatomical orthogonal coordinate system UVW

## Compute the change of basis matrix to go from image coordinate system to one locally defined bone coordinate system

def Image_to_bone_coordinate_system(image,U,V,W,bone_origin=None):

    nii = nib.load(image)

    M = np.identity(4)
    # Rotation bloc
    M[0][0] = U[0]
    M[1][0] = U[1]
    M[2][0] = U[2]
    M[0][1] = V[0]
    M[1][1] = V[1]
    M[2][1] = V[2]
    M[0][2] = W[0]
    M[1][2] = W[1]
    M[2][2] = W[2]

    # Translation bloc: here, origin coordinates are expressed in mm

    image_origin = sitk.ReadImage(image).GetOrigin() ## return image origin in mm

    if bone_origin is not None:

        #express translations in mm

        M[0][3] =  -bone_origin[0]*nii.header.get_zooms()[0] + image_origin[0]
        M[1][3] =  -bone_origin[1]*nii.header.get_zooms()[1] + image_origin[1]
        M[2][3] =  -bone_origin[2]*nii.header.get_zooms()[2] + image_origin[2]
   # else: return only rotations

    return(M)

#>>> Ti: 4*4 flirt transformation matrix expressed in the image coordinate system
#>>> M: 4*4 change of basis transformation matrix (from bone coordinate system to image coordinate system)


def Express_transformation_matrix_in_bone_coordinate_system(Ti,Mi):

    return np.dot(np.dot(Mi,Ti),inv(Mi))


def Binarize_fuzzy_mask(fuzzy_mask, binary_mask, threshold):

    nii = nib.load(fuzzy_mask)
    data = nii.get_data()
    output = np.zeros(data.shape)
    output[np.where(data>threshold)]= 1
    s = nib.Nifti1Image(output, nii.affine)
    nib.save(s,binary_mask)

    return 0


def Text_file_to_matrix(filename):
   T = np.loadtxt(str(filename), dtype='f')
   return np.mat(T)


def Matrix_to_text_file(matrix, text_filename):
    np.savetxt(text_filename, matrix, delimiter='  ')


def Rotation_vector_from_transformation_matrix(matrix):  ##### For more details see pages from 7 to 9: http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch2.pdf

    rotation_vector=np.zeros(3)
    rotation_vector[1]= -math.asin(matrix[0,2])
    c5= math.cos(rotation_vector[1])
    rotation_vector[1]= 180*rotation_vector[1]/math.pi
    rotation_vector[0]= 180*math.atan2((matrix[1,2] / c5),(matrix[2,2] / c5))/math.pi
    rotation_vector[2]= 180*math.atan2((matrix[0,1] / c5),(matrix[0,0] / c5))/math.pi

    return rotation_vector #return rotation vector in degrees [Rx Ry Rz]


def Translation_vector_from_transformation_matrix(matrix):

    return [matrix[0,3], matrix[1,3], matrix[2,3]] #return translation vector in mm [Tx Ty Tz]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--static', help='high-resolution static image', type=str, required = True)
    parser.add_argument('-dyn', '--dynamic', help='dynamic sequence', type=str, required = True)
    parser.add_argument('-ref', '--reference', help='reference time frame', type=int, required = True)
    parser.add_argument('-ref_path', '--ref_path', help='path to the folder containing the final results, of the high-resolution temporal reconstruction script. i.e. the folder containig the transformation matrices from static to each time frame.', type=str, required = True)
    parser.add_argument('-m', '--component', help='binary mask of the component in the first image in the low-resolution dynamic sequence', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)
    parser.add_argument('-exc', '--excel', help='Output Excel file', type=str, required = True)

    args = parser.parse_args()

    '''
    >>> Anatomically based coordinate system:
    >>> calcaneal coordinate system
    >>> the calcaneal x-axis(cx) was defined as the unit vector connecting the most anterior-inferior and the posterior-inferior
         calcaneal points, i.e. define the calcaneal x-axis (cx) from point0 and point1
    >>> In general the x-axis was directed to the left, the y-axis was directed superiorly and the z-axis was directly anteriorly.
    >>> the temporary calcaneal z-axis(∼cz) was defined as the unit vector connecting the insertion of the long plantar ligament
         (also known as the calcaneal origin,(Co) and the most convex point of the posterior-lateral curve of the calcaneus), i.e,
         define the calcaneal z-axis(cz) from point3 and point4
    >>> cross products were used to create orthogonal coordinate system. i.e, to determine the calcaneal y-axis (cy)
    >>> point1 : the most posterior-inferior calcaneal point
    >>> point2 : the most anterior-inferior calcaneal point
    >>> point3 : the calcaneal origin "Co", (see sheehan's paper for visual details)
    >>>
    '''

    nii = nib.load(args.static)

    '''####### Define the calcaneal coordinate system ##################'''

    ref_matrix_calcaneus = args.ref_path+'/propagation/output_path_component0/final_results/direct_static_on_dyn000'+str(args.reference-1)+'_component_0.mat'
    ref_matrix_talus = args.ref_path+'/propagation/output_path_component1/final_results/direct_static_on_dyn000'+str(args.reference-1)+'_component_1.mat'
    ref_matrix_tibia = args.ref_path+'/propagation/output_path_component2/final_results/direct_static_on_dyn000'+str(args.reference-1)+'_component_2.mat'

    #the calcaneal x-axis(cx) was defined as the unit vector connecting the most anterior-inferior and the posterior-inferior calcaneal point.

    ## point0 : the most posterior-inferior calcaneal point
    point0 = np.zeros([3,1])
    point0[0] = int(raw_input("Please enter the x_coordinate of the most posterior-inferior calcaneal point: "))-1
    point0[1] = int(raw_input("Please enter the y_coordinate of the most posterior-inferior calcaneal point: "))-1
    point0[2] = int(raw_input("Please enter the z_coordinate of the most posterior-inferior calcaneal point: "))-1
    ## map the point into the neutral position
    point0= warp_point_using_flirt_transform(point0,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))

    ## point1 : the most anterior-inferior calcaneal point
    point1 = np.zeros([3,1])
    point1[0] = int(raw_input("Please enter the x_coordinate of the most anterior-inferior calcaneal point: "))-1
    point1[1] = int(raw_input("Please enter the y_coordinate of the most anterior-inferior calcaneal point: "))-1
    point1[2] = int(raw_input("Please enter the z_coordinate of the most anterior-inferior calcaneal point: "))-1
    point1= warp_point_using_flirt_transform(point1,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))

    cx = (point1-point0)/LA.norm(point1-point0)

    ## point2 : the most convex point of the posterior-lateral curve of the calcaneus
    point2= np.zeros([3,1])
    point2[0] = int(raw_input("Please enter the x_coordinate of the most convex point of the posterior-lateral curve of the calcaneus: "))-1
    point2[1] = int(raw_input("Please enter the y_coordinate of the most convex point of the posterior-lateral curve of the calcaneus: "))-1
    point2[2] = int(raw_input("Please enter the z_coordinate of the most convex point of the posterior-lateral curve of the calcaneus: "))-1
    point2= warp_point_using_flirt_transform(point2,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))

    ## point3 : the insertion of the long plantar ligament (also known as the calcaneal origin,(Co))
    point3= np.zeros([3,1])
    point3[0] = int(raw_input("Please enter the x_coordinate of the insertion of the long plantar ligament: "))-1
    point3[1] = int(raw_input("Please enter the y_coordinate of the insertion of the long plantar ligament: "))-1
    point3[2] = int(raw_input("Please enter the z_coordinate of the insertion of the long plantar ligament : "))-1
    point3= warp_point_using_flirt_transform(point3,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))


    ###The temporary calcaneal z-axis(∼cz_t)was defined as the unit vector connecting the insertion of the long plantar ligament (also known as
    ###the calcaneal origin,(Co) and the most convex point of the posterior-lateral curve of the calcaneus.

    cz_t = (point3-point2)/LA.norm(point3-point2)

    # The cross product of cx and cz in R^3 is a vector perpendicular to both cx and cz
    ###For the calcaneus, cy was defined as(∼cz×cx)

    cy = np.cross(cz_t, cx, axis=0)
    cy/= LA.norm(cy)

    ###For the calcaneus, cz was defined as(cx×cy)

    cz = np.cross(cx,cy, axis=0)
    cz/= LA.norm(cz)

    '''####### Define the talar coordinate system ##################
    >>> Anatomically based coordinate system:
    >>> the talar x-axis(ax) was defined as the unit vector that bisected the arc formed by the two lines connecting the talar
    sinus with the most anterior-superior and anterior-inferior talar points.
    '''

    #talar sinus point
    point4 = np.zeros([3,1])
    point4[0] = int(raw_input("Please enter the x_coordinate of the talar sinus point: "))-1
    point4[1] = int(raw_input("Please enter the y_coordinate of the talar sinus point: "))-1
    point4[2] = int(raw_input("Please enter the z_coordinate of the talar sinus point: "))-1
    point4= warp_point_using_flirt_transform(point4,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

    #most anterior-superior talar point
    point5 = np.zeros([3,1])
    point5[0] = int(raw_input("Please enter the x_coordinate of the most anterior-superior talar point: "))-1
    point5[1] = int(raw_input("Please enter the y_coordinate of the most anterior-superior talar point: "))-1
    point5[2] = int(raw_input("Please enter the z_coordinate of the most anterior-superior talar point: "))-1
    point5= warp_point_using_flirt_transform(point5,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

    #most anterior-inferior talar point
    point6 = np.zeros([3,1])
    point6[0] = int(raw_input("Please enter the x_coordinate of the most anterior-inferior talar point: "))-1
    point6[1] = int(raw_input("Please enter the y_coordinate of the most anterior-inferior talar point: "))-1
    point6[2] = int(raw_input("Please enter the z_coordinate of the most anterior-inferior talar point: "))-1
    point6= warp_point_using_flirt_transform(point6,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))


    ax= (point5-point4)*LA.norm(point6-point4) + (point6-point4)*LA.norm(point5-point4)
    ax/= LA.norm(ax)
    #####################################################################################

    ##>>> the temporary talar y-axis(∼ay) was defined as the line that bisected the arc formed by the triangle defining the distal
    ##talar surface directly inferior to the talar dome. Ao was the most inferior point on the talar dome section of the talus.
    #Ao: the most inferior point on the talar dome section of the talus

    point7 = np.zeros([3,1])
    point7[0] = int(raw_input("Please enter the x_coordinate of the most inferior point on the talar dome section of the talus: "))-1
    point7[1] = int(raw_input("Please enter the y_coordinate of the most inferior point on the talar dome section of the talus: "))-1
    point7[2] = int(raw_input("Please enter the z_coordinate of the most inferior point on the talar dome section of the talus: "))-1
    point7= warp_point_using_flirt_transform(point7,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

    #Defining the distal talar surface directly inferior to the talar dome:

    #point A:
    point8 = np.zeros([3,1])
    point8[0] = int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the x_coordinate of the point A: "))-1
    point8[1] = int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the y_coordinate of the point A: "))-1
    point8[2] = int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the z_coordinate of the point A: "))-1
    point8= warp_point_using_flirt_transform(point8,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

    #point B:
    point9 = np.zeros([3,1])
    point9[0] = int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the x_coordinate of the point B: "))-1
    point9[1] = int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the y_coordinate of the point B: "))-1
    point9[2] = int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the z_coordinate of the point B: "))-1
    point9= warp_point_using_flirt_transform(point9,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

    ay_t= (point9-point7)*LA.norm(point8-point7) + (point8-point7)*LA.norm(point9-point7)
    ay_t/= LA.norm(ay_t)

    #>>> For the talus tz was defined as (∼ty×tx) and ty was defined as(tx×tz)

    # The cross product of ax and ay in R^3 is a vector perpendicular to both ax and ay

    az = np.cross(ax,ay_t,axis=0)
    az/= LA.norm(az)

    ay = np.cross(az, ax,axis=0)
    ay/= LA.norm(ay)

    '''####### Define the tibial coordinate system ##################
    >>> Anatomically based coordinate system:
    >>>  the tibial y-axis,(ty) was defined as the unit vector parallel to the tibial anterior edge in the sagittal-oblique image
    '''
    #point P1: inferior point from the the tibial anterior edge
    point10 = np.zeros([3,1])
    point10[0] = int(raw_input("please enter the x_coordinate of an inferior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
    point10[1] = int(raw_input("please enter the y_coordinate of an inferior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
    point10[2] = int(raw_input("please enter the z_coordinate of an inferior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
    point10= warp_point_using_flirt_transform(point10,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

    #point P2: superior point from the the tibial anterior edge

    point11 = np.zeros([3,1])
    point11[0] = int(raw_input("please enter the x_coordinate of a superior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
    point11[1] = int(raw_input("please enter the y_coordinate of a superior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
    point11[2] = int(raw_input("please enter the z_coordinate of a superior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
    point11= warp_point_using_flirt_transform(point11,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

    ty = (point11-point10)/LA.norm(point11-point10)

    #the temporary tibial z-axis(∼tz) was defined as the unit vector connecting the most lateral and medial tibial points

    #point P3: the most external lateral tibial point
    point12 = np.zeros([3,1])
    point12[0] = int(raw_input("please enter the x_coordinate of the most external lateral tibial point: "))-1
    point12[1] = int(raw_input("please enter the y_coordinate of the most external lateral tibial point: "))-1
    point12[2] = int(raw_input("please enter the z_coordinate of the most external lateral tibial point: "))-1
    point12= warp_point_using_flirt_transform(point12,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

    #point P3: the most internal lateral tibial point
    point13 = np.zeros([3,1])
    point13[0] = int(raw_input("please enter the x_coordinate of the most internal lateral tibial point: "))-1
    point13[1] = int(raw_input("please enter the y_coordinate of the most internal lateral tibial point: "))-1
    point13[2] = int(raw_input("please enter the z_coordinate of the most internal lateral tibial point: "))-1
    point13= warp_point_using_flirt_transform(point13,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

    tz_t = (point12-point13)/LA.norm(point12-point13)
    tz_t/= LA.norm(tz_t)

    tx = np.cross(ty, tz_t,axis=0)
    tx/= LA.norm(tx)

    tz = np.cross(tx, ty,axis=0)
    tz/= LA.norm(tz)

    '''
    >>> Define origins:

    Co: the calcaneal origin was defined as the insertion of the long plantar ligament

    Ao: the talar origin was defined as the most inferior point on the talar dome section of the talus

    To: The tibial origin was defined as the point that bisected ∼tz

    '''

    Co = point3
    Ao = point7
    To = (point12+point13)/2

    ### Compute the change of basis matrices

    M_calcaneus = Image_to_bone_coordinate_system(args.static,cx,cy,cz,bone_origin=Co)
    M_talus = Image_to_bone_coordinate_system(args.static,ax,ay,az,bone_origin=Ao)
    M_tibia = Image_to_bone_coordinate_system(args.static,tx,ty,tz,bone_origin=To)


    outputpath= args.output+'bone_motions/'
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    for i in range(0, len(args.component)):
        component_outputpath=outputpath+'component'+str(i)
        if not os.path.exists(component_outputpath):
            os.makedirs(component_outputpath)


    boneSet=glob.glob(outputpath+'/*')
    boneSet.sort()


    for i in range(0, len(args.component)):

        go = 'cp '+ args.component[i] +' '+boneSet[i]+'/mask_dyn0000_component'+str(i)+'.nii.gz'
        os.system(go)


    dynamic_basename = 'dyn'

    go = 'fslsplit ' + ' '+args.dynamic+' '+outputpath+dynamic_basename
    os.system(go)

    dynamicSet = glob.glob(outputpath+'/'+dynamic_basename+'*.nii.gz')
    dynamicSet.sort()



    bone_basename = 'mask_dyn'
    transform_basename = 'matrix'

    for bone in range (0, len(args.component)):


        for i in range(0, len(dynamicSet)-1):
            prefix = dynamicSet[i].split('/')[-1].split('.')[0]
            prefix1 = dynamicSet[i+1].split('/')[-1].split('.')[0]
            refweightSet = glob.glob(boneSet[bone]+'/'+bone_basename+'*.nii.gz')
            output_matrix = boneSet[bone]+'/'+'matrix_'+prefix+'_component'+str(bone)+'.mat'
            output_image = boneSet[bone]+'/'+'dyn'+str(i+1)+'_on_dyn_'+str(i)+'.nii.gz'
            refweightSet.sort()
            go= 'flirt -dof 6 -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -anglerep quaternion '
            go = go + ' -omat ' + output_matrix + ' -out  '+ output_image
            go = go + ' -in '+ dynamicSet[i+1] + ' -ref '+ dynamicSet[i]+ ' -refweight '+ refweightSet[i]
            print(go)
            os.system(go)

            Matrix_to_text_file(inv(Text_file_to_matrix(output_matrix)), output_matrix)

            input_mask = boneSet[bone]+'/'+'mask_'+prefix+'_component'+str(bone)+'.nii.gz'
            output_mask = boneSet[bone]+'/'+'mask_'+prefix1+'_component'+str(bone)+'.nii.gz'

            # Mask warping

            applyxfm(input_mask, dynamicSet[i+1], output_mask, Text_file_to_matrix(output_matrix), 'nearest')
            Binarize_fuzzy_mask(output_mask, output_mask, 0.5)

    change_of_basis_matrix = [M_calcaneus, M_talus, M_tibia]

    for bone in range (0, len(args.component)):

        image_matrixSet = glob.glob(outputpath+'/component'+str(bone)+'/'+'*.mat')
        image_matrixSet.sort()

        # create one excel file per bone
        book = xlwt.Workbook()
        # add new colour to palette and set RGB colour value
        xlwt.add_palette_colour("custom_colour", 0x29) #light_turquoise for forward motion
        style_f = xlwt.easyxf('pattern: pattern solid, fore_colour custom_colour')

        xlwt.add_palette_colour("new_custom_colour", 0x34) #light orange for backward motion
        style_b = xlwt.easyxf('pattern: pattern solid, fore_colour new_custom_colour')

        # create subfolders for joints of interest

        for c in range (len(change_of_basis_matrix)):

            new_folder = outputpath+'/component'+str(bone)+'/bone'+str(bone)+'_with_respect_to_bone_'+str(c)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            sheet = book.add_sheet('joint'+str(bone)+'__'+str(c))

            sheet.write(0,1,'Rx(deg)')
            sheet.write(0,2,'Ry(deg)')
            sheet.write(0,3,'Rz(deg)')
            sheet.write(0,4,'Tx(mm)')
            sheet.write(0,5,'Ty(mm)')
            sheet.write(0,6,'Tz(mm)')

            sheet.write(0,7,'cumsum Rx')
            sheet.write(0,8,'cumsum Ry')
            sheet.write(0,9,'cumsum Rz')
            sheet.write(0,10,'cumsum Tx')
            sheet.write(0,11,'cumsum Ty')
            sheet.write(0,12,'cumsum Tz')

            s_path = args.excel+'component'+str(bone)+'/component_'+str(bone)+'.xls'

            cumsum_Rx_backward = 0
            cumsum_Ry_backward = 0
            cumsum_Rz_backward = 0
            cumsum_Tx_backward = 0
            cumsum_Ty_backward = 0
            cumsum_Tz_backward = 0

            cumsum_Rx_forward = 0
            cumsum_Ry_forward = 0
            cumsum_Rz_forward = 0
            cumsum_Tx_forward = 0
            cumsum_Ty_forward = 0
            cumsum_Tz_forward = 0

            for i in range(0, len(dynamicSet)-1):
                prefix = dynamicSet[i].split('/')[-1].split('.')[0]
                joint = Express_transformation_matrix_in_bone_coordinate_system(inv(Text_file_to_matrix(image_matrixSet[i])),change_of_basis_matrix[c])
                save_path = new_folder+'/joint_kinematics'+prefix+'.mat'
                Matrix_to_text_file(joint, save_path)


                jo = Express_transformation_matrix_in_bone_coordinate_system(inv(Text_file_to_matrix(image_matrixSet[i])),change_of_basis_matrix[c])
                sheet.write(i+2,0,'tf'+str(i+1)+' on tf'+str(i+2),style_f)

                #rotations
                sheet.write(i+2,1,Rotation_vector_from_transformation_matrix(jo)[0],style_f)
                cumsum_Rx_forward += Rotation_vector_from_transformation_matrix(jo)[0]
                sheet.write(i+2,7,cumsum_Rx_forward,style_f)

                sheet.write(i+2,2,Rotation_vector_from_transformation_matrix(jo)[1],style_f)
                cumsum_Ry_forward += Rotation_vector_from_transformation_matrix(jo)[1]
                sheet.write(i+2,8,cumsum_Ry_forward,style_f)

                sheet.write(i+2,3,Rotation_vector_from_transformation_matrix(jo)[2],style_f)
                cumsum_Rz_forward += Rotation_vector_from_transformation_matrix(jo)[2]
                sheet.write(i+2,9,cumsum_Rz_forward,style_f)

                #translations
                sheet.write(i+2,4,Translation_vector_from_transformation_matrix(jo)[0],style_f)
                cumsum_Tx_forward += Translation_vector_from_transformation_matrix(jo)[0]
                sheet.write(i+2,10,cumsum_Tx_forward,style_f)

                sheet.write(i+2,5,Translation_vector_from_transformation_matrix(jo)[1],style_f)
                cumsum_Ty_forward += Translation_vector_from_transformation_matrix(jo)[1]
                sheet.write(i+2,11,cumsum_Ty_forward,style_f)

                sheet.write(i+2,6,Translation_vector_from_transformation_matrix(jo)[2],style_f)
                cumsum_Tz_forward += Translation_vector_from_transformation_matrix(jo)[2]
                sheet.write(i+2,12,cumsum_Tz_forward,style_f)

        book.save(s_path)
'''
                if (i < args.reference-1):

                    jo = Express_transformation_matrix_in_bone_coordinate_system(Text_file_to_matrix(image_matrixSet[args.reference-i-2]),change_of_basis_matrix[c])
                    sheet.write(i+2,0,'tf'+str(args.reference-i)+' on tf'+str(args.reference-i-1), style_b)

                    #rotations
                    sheet.write(i+2,1,Rotation_vector_from_transformation_matrix(jo)[0],style_b)
                    cumsum_Rx_backward += Rotation_vector_from_transformation_matrix(jo)[0]
                    sheet.write(i+2,7,cumsum_Rx_backward,style_b)


                    sheet.write(i+2,2,Rotation_vector_from_transformation_matrix(jo)[1],style_b)
                    cumsum_Ry_backward += Rotation_vector_from_transformation_matrix(jo)[1]
                    sheet.write(i+2,8,cumsum_Ry_backward,style_b)

                    sheet.write(i+2,3,Rotation_vector_from_transformation_matrix(jo)[2],style_b)
                    cumsum_Rz_backward += Rotation_vector_from_transformation_matrix(jo)[2]
                    sheet.write(i+2,9,cumsum_Rz_backward,style_b)

                    #translations
                    sheet.write(i+2,4,Translation_vector_from_transformation_matrix(jo)[0],style_b)
                    cumsum_Tx_backward += Translation_vector_from_transformation_matrix(jo)[0]
                    sheet.write(i+2,10,cumsum_Tx_backward,style_b)

                    sheet.write(i+2,5,Translation_vector_from_transformation_matrix(jo)[1],style_b)
                    cumsum_Ty_backward += Translation_vector_from_transformation_matrix(jo)[1]
                    sheet.write(i+2,11,cumsum_Ty_backward,style_b)

                    sheet.write(i+2,6,Translation_vector_from_transformation_matrix(jo)[2],style_b)
                    cumsum_Tz_backward += Translation_vector_from_transformation_matrix(jo)[2]
                    sheet.write(i+2,12,cumsum_Tz_backward,style_b)


                else:

                    jo = Express_transformation_matrix_in_bone_coordinate_system(inv(Text_file_to_matrix(image_matrixSet[i])),change_of_basis_matrix[c])
                    sheet.write(i+2,0,'tf'+str(i+1)+' on tf'+str(i+2),style_f)

                    #rotations
                    sheet.write(i+2,1,Rotation_vector_from_transformation_matrix(jo)[0],style_f)
                    cumsum_Rx_forward += Rotation_vector_from_transformation_matrix(jo)[0]
                    sheet.write(i+2,7,cumsum_Rx_forward,style_f)

                    sheet.write(i+2,2,Rotation_vector_from_transformation_matrix(jo)[1],style_f)
                    cumsum_Ry_forward += Rotation_vector_from_transformation_matrix(jo)[1]
                    sheet.write(i+2,8,cumsum_Ry_forward,style_f)

                    sheet.write(i+2,3,Rotation_vector_from_transformation_matrix(jo)[2],style_f)
                    cumsum_Rz_forward += Rotation_vector_from_transformation_matrix(jo)[2]
                    sheet.write(i+2,9,cumsum_Rz_forward,style_f)

                    #translations
                    sheet.write(i+2,4,Translation_vector_from_transformation_matrix(jo)[0],style_f)
                    cumsum_Tx_forward += Translation_vector_from_transformation_matrix(jo)[0]
                    sheet.write(i+2,10,cumsum_Tx_forward,style_f)

                    sheet.write(i+2,5,Translation_vector_from_transformation_matrix(jo)[1],style_f)
                    cumsum_Ty_forward += Translation_vector_from_transformation_matrix(jo)[1]
                    sheet.write(i+2,11,cumsum_Ty_forward,style_f)

                    sheet.write(i+2,6,Translation_vector_from_transformation_matrix(jo)[2],style_f)
                    cumsum_Tz_forward += Translation_vector_from_transformation_matrix(jo)[2]
                    sheet.write(i+2,12,cumsum_Tz_forward,style_f)


        book.save(s_path)
'''
