#!/usr/bin/env python2
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

import os
import os.path
import glob
import numpy as np
from numpy import matrix
from numpy.linalg import inv
import nibabel as nib
import nipype.algorithms.metrics as nipalg
import argparse
import multiprocessing




# Read from text file and store in matrix
#
# Parameters
# ----------
# filename : text filename (.mat)
#
# Returns matrix of floats
# -------
# output :
# 4*4 transformation matrix


def Text_file_to_matrix(filename):
   T = np.loadtxt(str(filename), dtype='f')
   return np.mat(T)

# save matrix as a text file
#
# Parameters
# ----------
# matrix : the matrix to be saved as text file
# text_filename : desired name of the text file (.mat)

def Matrix_to_text_file(matrix, text_filename):
    np.savetxt(text_filename, matrix, delimiter='  ')

# compute the dice score between two binary masks
#
# Parameters
# ----------
# rfile : reference image filename
# ifile : input image filename
#
# Returns
# -------
# output : scalar
# dice score between the two masks


def Bin_dice(rfile,ifile):

    tmp=nib.load(rfile) #segmentation
    d1=tmp.get_data()
    tmp=nib.load(ifile) #segmentation
    d2=tmp.get_data()
    d1_2=np.zeros(np.shape(tmp.get_data()))

    d1_2=len(np.where(d1*d2 != 0)[0])+0.0
    d1=len(np.where(d1 != 0)[0])+0.0
    d2=len(np.where(d2 != 0)[0])+0.0

    if d1==0:
        print ('ERROR: reference image is empty')
        dice=0
    elif d2==0:
        print ('ERROR: input image is empty')
        dice=0
    else:
        dice=2*d1_2/(d1+d2)

    return dice


def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())


# compute the dice score between two blurry masks
#
# Parameters
# ----------
# rfile : reference image filename
# ifile : input image filename
#
# Returns
# -------
# output : scalar
# overlap between the two blurry masks


def Fuzzy_dice(rfile, ifile):

    overlap = nipalg.FuzzyOverlap()
    overlap.inputs.in_ref = rfile
    overlap.inputs.in_tst = ifile
    overlap.inputs.weighting = 'volume'
    res = overlap.run()

    return res.outputs.dice

def Binarize_fuzzy_mask(fuzzy_mask, binary_mask, threshold):

    nii = nib.load(fuzzy_mask)
    data = nii.get_data()
    output = np.zeros(data.shape)
    output[np.where(data>threshold)]= 1
    s = nib.Nifti1Image(output, nii.affine)
    nib.save(s,binary_mask)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dynMRI')

    parser.add_argument('-s', '--static', help='Static input image', type=str, required = True)
    parser.add_argument('-d', '--dyn', help='Dynamic 4D input image', type=str, required = True)
    parser.add_argument('-m', '--mask', help='Segmentation high-resolution mask image', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)
    parser.add_argument('-os', '--OperatingSystem', help='Operating system: 0 if Linux and 1 if Mac Os', type=int, required = True)

    args = parser.parse_args()

    if (args.OperatingSystem == 0):
        call_flirt = 'fsl5.0-flirt'
        call_fslsplit = 'fsl5.0-fslsplit'

    elif (args.OperatingSystem == 1):
        call_flirt = 'flirt'
        call_fslsplit = 'fslsplit'
    else :
        print(" \n Please select your Operating System: 0 if Linux and 1 if Mac Os \n")


#######################Output path creation##########################

    outputpath=args.output
    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

############################## Data #################################

    High_resolution_static = args.static # 3D image
    dynamic = args.dyn  # 4D volume

############# split the 4D file into lots of 3D file s###############

    output_basename = 'dyn'
    go = call_fslsplit + ' '+dynamic+' '+outputpath+output_basename
    os.system(go)

############ Get the sorted set of 3D time frames ###################

    dynamicSet = glob.glob(outputpath+'/'+output_basename+'*.nii.gz')
    dynamicSet.sort()

################# Automated folders creation ########################

    outputpath_bone=outputpath+'propagation'
    if not os.path.exists(outputpath_bone):
        os.makedirs(outputpath_bone)


    for i in range(0, len(args.mask)):
        component_outputpath=outputpath_bone+'/output_path_component'+str(i)
        if not os.path.exists(component_outputpath):
            os.makedirs(component_outputpath)

    outputpath_boneSet=glob.glob(outputpath_bone+'/*')
    outputpath_boneSet.sort()

########################### Notations ################################
# t describes the time
# i describes the the component or the bone
######################################################################

####################### Define basenames #############################

    global_mask_basename = 'global_mask'
    global_image_basename = 'flirt_global_static'
    global_matrix_basename = 'global_static_on'
    local_matrix_basename = 'transform_dyn'
    mask_basename = 'mask_dyn'
    direct_transform_basename = 'direct_static_on'
    propagation_matrix_basename = 'matrix_flirt'
    propagation_image_basename = 'flirt_dyn'

######## Global registration of the static on each time frame #########

    movimage= High_resolution_static

    for t in range(0, len(dynamicSet)):

        refimage = dynamicSet[t]
        prefix = dynamicSet[t].split('/')[-1].split('.')[0]
        global_outputimage = outputpath+'flirt_global_static_on_'+prefix+'.nii.gz'
        global_outputmat = outputpath+'global_static_on_'+prefix+'.mat'
        go_init = call_flirt+' -noresampblur -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -cost mutualinfo  -dof 6 -ref '+refimage+' -in '+movimage+' -out '+global_outputimage+' -omat '+global_outputmat
        os.system(go_init)

#########################################################################

    global_matrixSet=glob.glob(outputpath+global_matrix_basename+'*.mat')
    global_matrixSet.sort()

    global_imageSet=glob.glob(outputpath+global_image_basename+'*.nii.gz')
    global_imageSet.sort()

##### Propagate manual segmentations into the low_resolution domain #####
############# using the estimated global transformations ################

    for i in range(0,len(outputpath_boneSet)):

        for t in range(0, len(global_imageSet)):

            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            global_mask= outputpath_boneSet[i]+'/global_mask_'+prefix+'_component_'+str(i)+'.nii.gz'
            go_propagation = call_flirt +' -applyxfm -noresampblur -ref '+global_imageSet[t]+' -in ' + args.mask[i] + ' -init '+ global_matrixSet[t] + ' -out ' + global_mask  + ' -interp nearestneighbour '
            os.system(go_propagation)
            Binarize_fuzzy_mask(global_mask, global_mask, 0.5)

##########################################################################

##############Local Rigid registration of bones###########################

    for i in range(0,len(outputpath_boneSet)):

        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()

        for t in range(0, len(dynamicSet)):

            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            refimage = dynamicSet[t]
            local_outputimage = outputpath_boneSet[i]+'/flirt_'+prefix+'_on_global_component_'+str(i)+'.nii.gz'
            local_outputmat = outputpath_boneSet[i]+'/transform_'+prefix+'_on_global_component_'+str(i)+'.mat'
            #go_init = 'flirt -searchrx -40 40 -searchry -40 40 -searchrz -40 40  -dof 6 -anglerep quaternion  -in '+refimage+' -ref '+global_imageSet[t]+' -out '+local_outputimage+' -omat '+local_outputmat +' -refweight '+ global_maskSet[t]
            go_init = call_flirt +' -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -dof 6 -in '+global_imageSet[t]+' -ref '+refimage+' -out '+local_outputimage+' -omat '+local_outputmat +' -inweight '+ global_maskSet[t]

###################### Talus registration ################################
            if(i==1):
                go_init = call_flirt +' -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -cost normcorr -dof 6 -in '+global_imageSet[t]+' -ref '+refimage+' -out '+local_outputimage+' -omat '+local_outputmat +' -inweight '+ global_maskSet[t]

            os.system(go_init)

#### Compute composed transformations from static to each time frame #####

    for i in range(0,len(outputpath_boneSet)):

        local_matrixSet = glob.glob(outputpath_boneSet[i]+'/'+local_matrix_basename+'*.mat')
        local_matrixSet.sort()

        for t in range(0, len(dynamicSet)):

            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            #static_to_dyn_matrix = np.dot(np.linalg.inv(Text_file_to_matrix(local_matrixSet[t])), Text_file_to_matrix(global_matrixSet[t]))
            static_to_dyn_matrix = np.dot(Text_file_to_matrix(local_matrixSet[t]), Text_file_to_matrix(global_matrixSet[t]))

            save_path= outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.mat'
            np.savetxt(save_path,static_to_dyn_matrix, delimiter='  ')

###### Propagate the high_resolution segmentations into time frames #######
####### using the estimated composed transformations from static ##########
########################## to each time frame #############################

    for i in range(0,len(outputpath_boneSet)):

        init_matrixSet = glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.mat')
        init_matrixSet.sort()

        for t in range(0, len(dynamicSet)):

            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            low_resolution_mask = outputpath_boneSet[i]+'/mask_'+prefix+'_component_'+str(i)+'.nii.gz'
            go_init = call_flirt + ' -applyxfm -noresampblur -ref '+ dynamicSet[t] + ' -in '+ args.mask[i] + ' -out '+ low_resolution_mask + ' -init '+init_matrixSet[t]+ ' -interp nearestneighbour '
            os.system(go_init)
            Binarize_fuzzy_mask(low_resolution_mask, low_resolution_mask, 0.5)

######### Finding the time frame that best align with static image  ########

    dice_evaluation_array=np.zeros((len(outputpath_boneSet), len(dynamicSet)))

    #Hausdroff_evaluation_array=np.zeros((len(outputpath_boneSet), len(dynamicSet)))

    for i in range(0,len(outputpath_boneSet)):

        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()
        maskSet=glob.glob(outputpath_boneSet[i]+'/'+mask_basename+'*.nii.gz')
        maskSet.sort()
        direct_static_on_dynSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.nii.gz')
        direct_static_on_dynSet.sort()

        for t in range(0, len(dynamicSet)):

            dice_evaluation_array[i][t] = Bin_dice(maskSet[t],global_maskSet[t])
            #Hausdroff_evaluation_array[i][t] = directed_hausdorff(nifti_to_array(maskSet[t]),nifti_to_array(global_maskSet[t]))[0]

    linked_time_frame=np.prod(dice_evaluation_array, axis=0)
    #linked_time_frame=np.prod(Hausdroff_evaluation_array, axis=0)
    t=np.argmax(linked_time_frame) ### the main idea here is to detect the time frame the most closest to the static scan ("no-motion" detection)
    #t=np.argmin(linked_time_frame) ### the main idea here is to detect the time frame the most closest to the static scan ("no-motion" detection)

########## copy the best "static/dynamic" registration results to the final outputs folder ############

    for i in range(0,len(outputpath_boneSet)):


        maskSet=glob.glob(outputpath_boneSet[i]+'/'+mask_basename+'*.nii.gz')
        maskSet.sort()
        transformationSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.mat')
        transformationSet.sort()

        output_results=outputpath_boneSet[i]+'/final_results'
        if not os.path.exists(output_results):
            os.makedirs(output_results)

        t=np.argmax(linked_time_frame)
        #t=np.argmin(linked_time_frame)


        copy= 'cp '+maskSet[t]+ '  '+ output_results
        os.system(copy) ######copy mask(t) to the final_results folder
        copy= 'cp '+transformationSet[t]+ '  '+ output_results
        os.system(copy) ######copy the most accurate estimated transformation in the "final_results" folder

        direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
        direct_static_on_dynSet.sort()
        final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
        final_refweightSet.sort()

######################## Backward propagation ##############################

        while(t>0):

            final_refweightSet.sort()
            movimage = dynamicSet[t-1]
            refimage = dynamicSet[t]
            go_init = call_flirt + ' -searchrx -40 40 -searchry -40 40  -searchrz -40 40  -anglerep quaternion  -dof 6 -ref '+refimage

            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t-1].split('/')[-1].split('.')[0]
            outputimage = output_results+'/flirt_'+prefix2+'_on_'+prefix1+'.nii.gz'
            outputmat   = output_results+'/matrix_flirt_'+prefix2+'_on_'+prefix1+'.mat'
            go = go_init +' -in '+movimage+' -out '+outputimage+ ' -omat '+outputmat + ' -refweight ' +  final_refweightSet[0]
            os.system(go)
            final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
            final_refweightSet.sort()
            direct_transform = output_results+'/direct_static_on_'+prefix2+'_component_'+str(i)+'.mat'
            direct_static_on_dyn=np.dot(inv(Text_file_to_matrix(outputmat)), Text_file_to_matrix(direct_static_on_dynSet[0]))
            Matrix_to_text_file(direct_static_on_dyn, direct_transform)
            direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
            direct_static_on_dynSet.sort()
            out_refweight= output_results+'/mask_'+prefix2+'_component_'+str(i)+'.nii.gz'
            go_propagation = call_flirt + ' -applyxfm -noresampblur -ref '+dynamicSet[t-1]+' -in ' + args.mask[i] + ' -init '+ direct_static_on_dynSet[0] + ' -out ' +out_refweight  + ' -interp nearestneighbour '
            os.system(go_propagation)

            t-=1


######################### Forward propagation ##############################

        direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
        direct_static_on_dynSet.sort()
        final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
        final_refweightSet.sort()

        t=np.argmax(linked_time_frame) #temporal position of the best aligned time frame with the static
        #t=np.argmin(linked_time_frame)

        direct_static_on_dynSet.sort()
        final_refweightSet.sort()

        while(t<len(dynamicSet)-1):

            final_refweightSet.sort()
            movimage = dynamicSet[t+1]
            refimage = dynamicSet[t]
            go_init = call_flirt + ' -searchrx -40 40 -searchry -40 40  -searchrz -40 40  -anglerep quaternion  -dof 6 -ref '+refimage
            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t+1].split('/')[-1].split('.')[0]
            outputimage = output_results+'/flirt_'+prefix2+'_on_'+prefix1+'.nii.gz'
            outputmat   = output_results+'/matrix_flirt_'+prefix2+'_on_'+prefix1+'.mat'
            go = go_init +' -in '+movimage+' -out '+outputimage+ ' -omat '+outputmat + ' -refweight ' +  final_refweightSet[t]
            os.system(go)
            direct_transform = output_results+'/direct_static_on_'+prefix2+'_component_'+str(i)+'.mat'
            direct_static_on_dyn=np.dot(inv(Text_file_to_matrix(outputmat)), Text_file_to_matrix(direct_static_on_dynSet[t]))
            Matrix_to_text_file(direct_static_on_dyn, direct_transform)
            direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
            direct_static_on_dynSet.sort()
            out_refweight= output_results+'/mask_'+prefix2+'_component_'+str(i)+'.nii.gz'
            go_propagation = call_flirt + ' -applyxfm -noresampblur -ref '+dynamicSet[t+1]+' -in ' + args.mask[i] + ' -init '+ direct_static_on_dynSet[t+1] + ' -out ' +out_refweight  + ' -interp nearestneighbour '
            os.system(go_propagation)
            final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
            final_refweightSet.sort()

            t+=1


########## Clean "final_results" folders inorder to only keep the low_resolution automated segmentations and the direct transformations from the static scan to each low-resolution time frame ########################
####################################  move all intermediate results from the "final_results" folder to the "outputpath_boneSet[i]" folder #############################################################################

    for i in range(0,len(outputpath_boneSet)):


        output_results=outputpath_boneSet[i]+'/final_results'

        propagation_matrixSet=glob.glob(output_results+'/'+propagation_matrix_basename+'*.mat')
        propagation_matrixSet.sort()

        propagation_imageSet=glob.glob(output_results+'/'+propagation_image_basename+'*.nii.gz')
        propagation_imageSet.sort()

        for t in range(0, len(propagation_matrixSet)):

            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t+1].split('/')[-1].split('.')[0]

            go1= 'mv '+ propagation_matrixSet[t] +' '+ outputpath_boneSet[i]+'/matrix_flirt_'+prefix1+'_on_'+prefix2+'.mat'
            go2= 'mv '+ propagation_imageSet[t] +' '+ outputpath_boneSet[i]+'/flirt_'+prefix1+'_on_'+prefix2+'.nii.gz'
            os.system(go1)
            os.system(go2)

    print(np.argmax(linked_time_frame))
