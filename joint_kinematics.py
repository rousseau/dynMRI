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


import glob
import numpy as np
import os
from numpy.linalg import inv
from numpy.linalg import det
import xlwt
from xlwt import Workbook
import math
import argparse
import nibabel as nib
from scipy.ndimage.interpolation import map_coordinates


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


def Binarize_fuzzy_mask(mask, threshold):

    nii = nib.load(mask)
    data = nii.get_data()
    binary_mask = np.zeros(data.shape)
    binary_mask[np.where(data>threshold)]= 1
    s = nib.Nifti1Image(binary_mask, nii.affine)
    nib.save(s,mask)

    return 0


def Text_file_to_matrix(filename):
   T = np.loadtxt(str(filename), dtype='f')
   return np.mat(T)


def Matrix_to_text_file(matrix, text_filename):
    np.savetxt(text_filename, matrix, delimiter='  ')


def Rotation_vector_from_transformation_matrix(matrix):  ##### For more details see pages from 7 to 9: http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch2.pdf
    s5=matrix[0,2]
    r12=matrix[0,1]
    r11=matrix[0,0]
    r23=matrix[1,2]
    r33=matrix[2,2]
    rotation_vector=np.zeros(3)
    rotation_vector[1]= -math.asin(s5)
    c5= math.cos(rotation_vector[1])
    rotation_vector[0]= math.atan2((r23 / c5),(r33 / c5))
    rotation_vector[2]= math.atan2((r12 / c5),(r11 / c5))

    rotation_vector[0]= (180*rotation_vector[0])/math.pi
    rotation_vector[1]= (180*rotation_vector[1])/math.pi
    rotation_vector[2]= (180*rotation_vector[2])/math.pi

    #np.set_printoptions(precision=6, suppress=True)
    return rotation_vector #return rotation vector in degrees [Rx Ry Rz]

def Translation_vector_from_transformation_matrix(matrix):

    translation_vector=np.zeros(3)
    translation_vector[0]=matrix[0,3]
    translation_vector[1]=matrix[1,3]
    translation_vector[2]=matrix[2,3]
    return(translation_vector) #return translation vector in mm [Tx Ty Tz]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dyn', '--dynamic', help='dynamic sequence', type=str, required = True)
    parser.add_argument('-m', '--component', help='binary mask of the component in the first image in the low-resolution dynamic sequence', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)

    args = parser.parse_args()

    outputpath= args.output+'bone_motions/'
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    for i in range(0, len(args.component)):
        component_outputpath=outputpath+'component'+str(i)
        if not os.path.exists(component_outputpath):
            os.makedirs(component_outputpath)

    #Create folder for subtalar (calcaneal-talar) joint

    subtalar_outputpath = args.output+'calcaneal_talar_joint'
    if not os.path.exists(subtalar_outputpath):
        os.makedirs(subtalar_outputpath)


    #Create folder for calcaneal-tibial complex

    calcaneal_tibial_outputpath = args.output+'calcaneal_tibial_complex'
    if not os.path.exists(calcaneal_tibial_outputpath):
        os.makedirs(calcaneal_tibial_outputpath)



    #Create folder for talocrural (talar-tibial) joint

    talocrural_outputpath = args.output+'talar_tibial_joint'
    if not os.path.exists(talocrural_outputpath):
        os.makedirs(talocrural_outputpath)


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
            Binarize_fuzzy_mask(output_mask, 0.5)

    ### Excel file creation for individual bone motions

    book = xlwt.Workbook()

    for bone in range (0, len(args.component)):

        sheet = book.add_sheet('component'+str(bone))
        sheet.write(0,1,'Rx(deg)')
        sheet.write(0,2,'Ry(deg)')
        sheet.write(0,3,'Rz(deg)')
        sheet.write(0,4,'Tx(mm)')
        sheet.write(0,5,'Ty(mm)')
        sheet.write(0,6,'Tz(mm)')

        transformSet = glob.glob(boneSet[bone]+'/'+transform_basename+'*.mat')
        transformSet.sort()


        for t in range(0, len(transformSet)):
            sheet.write(t+1,0,'time '+str(t))
            sheet.write(t+1,1,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(transformSet[t]))[0])
            sheet.write(t+1,2,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(transformSet[t]))[1])
            sheet.write(t+1,3,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(transformSet[t]))[2])

            sheet.write(t+1,4,Translation_vector_from_transformation_matrix(Text_file_to_matrix(transformSet[t]))[0])
            sheet.write(t+1,5,Translation_vector_from_transformation_matrix(Text_file_to_matrix(transformSet[t]))[1])
            sheet.write(t+1,6,Translation_vector_from_transformation_matrix(Text_file_to_matrix(transformSet[t]))[2])


    book.save(outputpath+'motions.xlsx')


    calcaneal_transformSet =  glob.glob(outputpath+'component0/'+transform_basename+'*.mat')
    calcaneal_transformSet.sort()

    talar_transformSet =  glob.glob(outputpath+'component1/'+transform_basename+'*.mat')
    talar_transformSet.sort()

    tibial_transformSet =  glob.glob(outputpath+'component2/'+transform_basename+'*.mat')
    tibial_transformSet.sort()



    for t in range(0, len(calcaneal_transformSet)):
        prefix = dynamicSet[t].split('/')[-1].split('.')[0]

        #Subtalar (calcaneal-talar) joint
        subtalar_matrix = subtalar_outputpath+'/'+'matrix_subtalar_'+prefix+'.mat'
        Matrix_to_text_file(np.dot(Text_file_to_matrix(calcaneal_transformSet[t]) , Text_file_to_matrix(talar_transformSet[t])), subtalar_matrix)

        #Calcaneal-tibial complex
        Calcaneal_tibial_matrix = calcaneal_tibial_outputpath+'/'+'matrix_calcaneal_tibial_'+prefix+'.mat'
        Matrix_to_text_file(np.dot(Text_file_to_matrix(calcaneal_transformSet[t]) , Text_file_to_matrix(tibial_transformSet[t])), Calcaneal_tibial_matrix)

        #Talocrural (talar-tibial) joint
        talocrural_matrix = talocrural_outputpath+'/'+'matrix_talocrural_'+prefix+'.mat'
        Matrix_to_text_file(np.dot(Text_file_to_matrix(talar_transformSet[t]) , Text_file_to_matrix(tibial_transformSet[t])), talocrural_matrix)


    subtalar_matrixSet = glob.glob(subtalar_outputpath+'/'+transform_basename+'*.mat')
    subtalar_matrixSet.sort()

    Calcaneal_tibial_matrixSet = glob.glob(calcaneal_tibial_outputpath+'/'+transform_basename+'*.mat')
    Calcaneal_tibial_matrixSet.sort()

    talocrural_matrixSet = glob.glob(talocrural_outputpath+'/'+transform_basename+'*.mat')
    talocrural_matrixSet.sort()


### Excel file creation for coupled bone motions

    book1 = xlwt.Workbook()

    sheet1 = book1.add_sheet('Subtalar joint')
    sheet1.write(0,1,'Rx(deg)')
    sheet1.write(0,2,'Ry(deg)')
    sheet1.write(0,3,'Rz(deg)')
    sheet1.write(0,4,'Tx(mm)')
    sheet1.write(0,5,'Ty(mm)')
    sheet1.write(0,6,'Tz(mm)')

    sheet2 = book1.add_sheet('Calcaneal-tibial complex')
    sheet2.write(0,1,'Rx(deg)')
    sheet2.write(0,2,'Ry(deg)')
    sheet2.write(0,3,'Rz(deg)')
    sheet2.write(0,4,'Tx(mm)')
    sheet2.write(0,5,'Ty(mm)')
    sheet2.write(0,6,'Tz(mm)')

    sheet3 = book1.add_sheet('Talocrural joint')
    sheet3.write(0,1,'Rx(deg)')
    sheet3.write(0,2,'Ry(deg)')
    sheet3.write(0,3,'Rz(deg)')
    sheet3.write(0,4,'Tx(mm)')
    sheet3.write(0,5,'Ty(mm)')
    sheet3.write(0,6,'Tz(mm)')


    for t in range(0, len(subtalar_matrixSet)):

        sheet1.write(t+1,0,'time '+str(t))
        sheet1.write(t+1,1,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(subtalar_matrixSet[t]))[0])
        sheet1.write(t+1,2,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(subtalar_matrixSet[t]))[1])
        sheet1.write(t+1,3,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(subtalar_matrixSet[t]))[2])

        sheet1.write(t+1,4,Translation_vector_from_transformation_matrix(Text_file_to_matrix(subtalar_matrixSet[t]))[0])
        sheet1.write(t+1,5,Translation_vector_from_transformation_matrix(Text_file_to_matrix(subtalar_matrixSet[t]))[1])
        sheet1.write(t+1,6,Translation_vector_from_transformation_matrix(Text_file_to_matrix(subtalar_matrixSet[t]))[2])

        sheet2.write(t+1,0,'time '+str(t))
        sheet2.write(t+1,1,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(Calcaneal_tibial_matrixSet[t]))[0])
        sheet2.write(t+1,2,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(Calcaneal_tibial_matrixSet[t]))[1])
        sheet2.write(t+1,3,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(Calcaneal_tibial_matrixSet[t]))[2])

        sheet2.write(t+1,4,Translation_vector_from_transformation_matrix(Text_file_to_matrix(Calcaneal_tibial_matrixSet[t]))[0])
        sheet2.write(t+1,5,Translation_vector_from_transformation_matrix(Text_file_to_matrix(Calcaneal_tibial_matrixSet[t]))[1])
        sheet2.write(t+1,6,Translation_vector_from_transformation_matrix(Text_file_to_matrix(Calcaneal_tibial_matrixSet[t]))[2])

        sheet3.write(t+1,0,'time '+str(t))
        sheet3.write(t+1,1,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(talocrural_matrixSet[t]))[0])
        sheet3.write(t+1,2,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(talocrural_matrixSet[t]))[1])
        sheet3.write(t+1,3,Rotation_vector_from_transformation_matrix(Text_file_to_matrix(talocrural_matrixSet[t]))[2])

        sheet3.write(t+1,4,Translation_vector_from_transformation_matrix(Text_file_to_matrix(talocrural_matrixSet[t]))[0])
        sheet3.write(t+1,5,Translation_vector_from_transformation_matrix(Text_file_to_matrix(talocrural_matrixSet[t]))[1])
        sheet3.write(t+1,6,Translation_vector_from_transformation_matrix(Text_file_to_matrix(talocrural_matrixSet[t]))[2])

    book1.save(outputpath+'kinematics.xlsx')
