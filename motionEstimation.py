#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

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
import matplotlib.pyplot as plt
from pylab import *
import nipype.algorithms.metrics as nipalg
import argparse



def karimText_file_to_matrix(filename):
   T = np.loadtxt(str(filename), dtype='f')
   return np.mat(T)
#
def karimMatrix_to_text_file(matrix, text_filename):
    np.savetxt(text_filename, matrix, delimiter='  ')
#
def KarimBin_dice(rfile,ifile):

    tmp=nib.load(rfile) #segmentation
    d1=tmp.get_data()
    tmp=nib.load(ifile) #segmentation
    d2=tmp.get_data()
    d1_2=np.zeros(np.shape(tmp.get_data()))

    d1_2=len(np.where(d1*d2==1)[0])+0.0
    d1=len(np.where(d1==1)[0])+0.0
    d2=len(np.where(d2==1)[0])+0.0
#
    if d1==0:
        print ('ERROR: reference image is empty')
        dice=0
    elif d2==0:
        print ('ERROR: input image is empty')
        dice=0
    else:
        dice=2*d1_2/(d1+d2)

    return dice
#
def KarimFuzzy_dice(ground_truth, input_image):
    overlap = nipalg.FuzzyOverlap()
    overlap.inputs.in_ref = ground_truth
    overlap.inputs.in_tst = input_image
    overlap.inputs.weighting = 'volume'
    res = overlap.run()

    return res.outputs.dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dynMRI')

    parser.add_argument('-s', '--static', help='Static input image', type=str, required = True)
    parser.add_argument('-d', '--dyn', help='Dynamic 4D input image', type=str, required = True)
    parser.add_argument('-m', '--mask', help='Segmentation high-resolution mask image', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)

    args = parser.parse_args()


##############Output path creation###################################

    outputpath=args.output
    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

##############Data###################################
    High_resolution_static = args.static #image 3D
    dynamic = args.dyn  #image 4D

#############Split 4D image into a set of 3D image using fsl5.0-fslsplit##############
    output_basename = 'dyn'
    go = 'fsl5.0-fslsplit '+dynamic+' '+outputpath+output_basename
    print(go)
    os.system(go)

    #Get the set of 3D time frames
    dynamicSet = glob.glob(outputpath+'/'+output_basename+'*.nii.gz')
    dynamicSet.sort()

#
#    ###### Automated folders creation ##############
#
    outputpath_bone=outputpath+'propagation'
    if not os.path.exists(outputpath_bone):
        os.makedirs(outputpath_bone)


    for i in range(0, len(args.mask)):
        component_outputpath=outputpath_bone+'/output_path_component'+str(i)
        if not os.path.exists(component_outputpath):
            os.makedirs(component_outputpath)

    outputpath_boneSet=glob.glob(outputpath_bone+'/*')
    outputpath_boneSet.sort()

#

    global_mask_basename = 'global_mask'
    global_image_basename = 'flirt_global_static'
    global_matrix_basename = 'global_static_on'
    mask_basename = 'mask_dyn'
    direct_transform_basename='direct_static_on'

#    ##############################################Global registration of the static on each time frame ############################################################################
    for t in range(0, len(dynamicSet)):

                refimage = dynamicSet[t]
                movimage= High_resolution_static
                prefix = dynamicSet[t].split('/')[-1].split('.')[0]
                global_outputimage = outputpath+'flirt_global_static_on_'+prefix+'.nii.gz'
                global_outputmat = outputpath+'global_static_on_'+prefix+'.mat'
#             ####Global rigid registration of dyn0 on static
                go_init = 'time flirt -noresampblur -searchrx -40 40 -searchry -40 40 -searchrz -40 40  -dof 6 -ref '+refimage+' -in '+movimage+' -out '+global_outputimage+' -omat '+global_outputmat
                print(go_init)
                os.system(go_init)
                for i in range(0,len(outputpath_boneSet)):
                    global_mask= outputpath_boneSet[i]+'/global_mask_'+prefix+'_component_'+str(i)+'.nii.gz'
                    go_propagation = 'time flirt -applyxfm -noresampblur -ref '+global_outputimage+' -in ' + args.mask[i] + ' -init '+ global_outputmat + ' -out ' + global_mask + ' -interp nearestneighbour '
                    print(go_propagation)
                    os.system(go_propagation)
#
    global_matrixSet=glob.glob(outputpath+global_matrix_basename+'*.mat')
    global_matrixSet.sort()

    global_imageSet=glob.glob(outputpath+global_image_basename+'*.nii.gz')
    global_imageSet.sort()

#    #
#    ####Local Rigid registration of bones########
#
    for i in range(0,len(outputpath_boneSet)):
#
        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()
        movimage= High_resolution_static
#
        for t in range(0, len(dynamicSet)):
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            refimage = dynamicSet[t]
            local_outputimage = outputpath_boneSet[i]+'/flirt_'+prefix+'_on_global.nii.gz'
            local_outputmat = outputpath_boneSet[i]+'/transform_'+prefix+'_on_global.mat'
            go_init = 'time flirt -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -dof 6 -in '+refimage+' -ref '+global_imageSet[t]+' -out '+local_outputimage+' -omat '+local_outputmat +' -refweight '+ global_maskSet[t]
            if(i==2):
                go_init = 'time flirt  -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -dof 6 -in '+global_imageSet[t]+' -ref '+refimage+' -out '+local_outputimage+' -omat '+local_outputmat +' -inweight '+ global_maskSet[t]

            print(go_init)
            os.system(go_init)

            global_on_dynN=np.linalg.inv(karimText_file_to_matrix(local_outputmat))

            if(i==2):
                global_on_dynN=karimText_file_to_matrix(local_outputmat)

            save_path_global=outputpath_boneSet[i]+'/static_on_'+prefix+'.mat'
            np.savetxt(save_path_global, global_on_dynN, delimiter=' ')
            link_matrix=np.dot(global_on_dynN, karimText_file_to_matrix(global_matrixSet[t]))
            save_path= outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.mat'
            np.savetxt(save_path,link_matrix, delimiter='  ')
            out2= outputpath_boneSet[i]+'/mask_'+prefix+'_component_'+str(i)+'.nii.gz'
            init=outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.mat'
            static_on_dyn0= outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.nii.gz'
            go_init = 'time flirt -applyxfm -noresampblur -ref '+ dynamicSet[t] + ' -in '+ args.mask[i] + ' -out '+ out2 + ' -init '+init+ ' -interp nearestneighbour '
            print(go_init)
            os.system(go_init)
        ############# write static_on_dyn0 3D image  ###########
            go = 'time flirt -applyxfm -noresampblur -ref '+ dynamicSet[t] + ' -in '+ movimage + ' -out '+ static_on_dyn0 + ' -init '+init
            print(go)
            os.system(go)

    ################ finding the time frame that best align with static image  #####################################################################################################################

    dice_evaluation_array=np.zeros((len(outputpath_boneSet), len(dynamicSet)))

    for i in range(0,len(outputpath_boneSet)):

        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()
        maskSet=glob.glob(outputpath_boneSet[i]+'/'+mask_basename+'*.nii.gz')
        maskSet.sort()
        for t in range(0, len(dynamicSet)):
            dice_evaluation_array[i][t] = KarimBin_dice(maskSet[t],global_maskSet[t])

    linked_time_frame=np.prod(dice_evaluation_array, axis=0)


    ############### Propagate segmentations in the Low-resolution domain ####################################

    for i in range(0,len(outputpath_boneSet)):
        ###create a folder to save final results#######
        maskSet=glob.glob(outputpath_boneSet[i]+'/'+mask_basename+'*.nii.gz')
        maskSet.sort()
        transformationSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.mat')
        transformationSet.sort()

        output_results=outputpath_boneSet[i]+'/final_results'
        if not os.path.exists(output_results):
            os.makedirs(output_results)

        t=np.argmax(linked_time_frame) #temporal position of the best aligned time frame with the static

        copy= 'cp '+maskSet[t]+ '  '+ output_results
        os.system(copy) ######copy mask(t) to the final_results folder
        copy= 'cp '+transformationSet[t]+ '  '+ output_results
        os.system(copy) ######copy the transformation that links between static and dynamic data
        direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
        direct_static_on_dynSet.sort()
        final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
        final_refweightSet.sort()

    ########Backward propagation ###############################

        while(t>0):

            final_refweightSet.sort()
            movimage = dynamicSet[t-1]
            refimage = dynamicSet[t]
            go_init = 'time flirt -searchrx -40 40 -searchry -40 40  -searchrz -40 40 -anglerep quaternion  -dof 6 -ref '+refimage
            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t-1].split('/')[-1].split('.')[0]
            outputimage = output_results+'/flirt_'+prefix2+'_on_'+prefix1+'.nii.gz'
            outputmat   = output_results+'/matrix_flirt_'+prefix2+'_on_'+prefix1+'.mat'
            go = go_init +' -in '+movimage+' -out '+outputimage+ ' -omat '+outputmat + ' -refweight ' +  final_refweightSet[0]
            print(go)
            os.system(go)
            final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
            final_refweightSet.sort()
            direct_transform = output_results+'/direct_static_on_'+prefix2+'_component_'+str(i)+'.mat'
            b=np.dot(inv(karimText_file_to_matrix(outputmat)), karimText_file_to_matrix(direct_static_on_dynSet[0]))
            karimMatrix_to_text_file(b, direct_transform)
            direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
            direct_static_on_dynSet.sort()
            out_refweight= output_results+'/mask_'+prefix2+'_component_'+str(i)+'.nii.gz'
            go_propagation = 'time flirt -applyxfm -noresampblur -ref '+dynamicSet[t-1]+' -in ' + args.mask[i] + ' -init '+ direct_static_on_dynSet[0] + ' -out ' +out_refweight  + ' -interp nearestneighbour '
            print(go_propagation)
            os.system(go_propagation)

            t-=1


    ########Forward propagation ###############################
        direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
        direct_static_on_dynSet.sort()
        final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
        final_refweightSet.sort()

        t=np.argmax(linked_time_frame) #temporal position of the best aligned time frame with the static

        direct_static_on_dynSet.sort()
        final_refweightSet.sort()

        while(t<len(dynamicSet)-1):

            final_refweightSet.sort()
            print(final_refweightSet)
            movimage = dynamicSet[t+1]
            refimage = dynamicSet[t]
            go_init = 'time flirt -searchrx -40 40 -searchry -40 40  -searchrz -40 40 -anglerep quaternion  -dof 6 -ref '+refimage
            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t+1].split('/')[-1].split('.')[0]
            outputimage = output_results+'/flirt_'+prefix2+'_on_'+prefix1+'.nii.gz'
            outputmat   = output_results+'/matrix_flirt_'+prefix2+'_on_'+prefix1+'.mat'
            go = go_init +' -in '+movimage+' -out '+outputimage+ ' -omat '+outputmat + ' -refweight ' +  final_refweightSet[t]
            print(go)
            os.system(go)
            direct_transform = output_results+'/direct_static_on_'+prefix2+'_component_'+str(i)+'.mat'
            b=np.dot(inv(karimText_file_to_matrix(outputmat)), karimText_file_to_matrix(direct_static_on_dynSet[t]))
            karimMatrix_to_text_file(b, direct_transform)
            direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
            direct_static_on_dynSet.sort()
            out_refweight= output_results+'/mask_'+prefix2+'_component_'+str(i)+'.nii.gz'
            go_propagation = 'time flirt -applyxfm -noresampblur -ref '+dynamicSet[t+1]+' -in ' + args.mask[i] + ' -init '+ direct_static_on_dynSet[t+1] + ' -out ' +out_refweight  + ' -interp nearestneighbour '
            print(go_propagation)
            os.system(go_propagation)
            final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
            final_refweightSet.sort()

            t+=1




    print(linked_time_frame)
    print("The best aligned time frame with static is:\n")
    print(np.argmax(linked_time_frame))
    plt.plot(linked_time_frame)
    xlabel('time')
    ylabel('Dice product')
    title('subject5')
    save_figure_path=args.output+'dice_product.png'
    plt.show()
