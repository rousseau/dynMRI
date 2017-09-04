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
import multiprocessing


#@author: Karim


# compute the normalized correlation between image 1 and image 2 using binary mask
#
# Parameters
# ----------
# image1 : nifti file name of image 1
# image 2: nifti file name of image 2
# mask: nifti file name of the binary mask
#
# Returns
# -------
# output : scalar
# normalized correlation between image 1 and image 2 over all the region of interest described by the binary mask


def Normalized_cross_Correlation(image1, image2, mask):

    nii1 = nib.load(image1)
    im_gnd1= nii1.get_data()  # Data voxel here (3D) as an array
    nii2 = nib.load(image2)
    im_gnd2= nii2.get_data()
    nii3 = nib.load(mask)
    im_gnd3= nii3.get_data()

    naxes0, naxes1, naxes2 = im_gnd1.shape
    Cropped_image1=np.zeros((naxes0,naxes1,naxes2))
    Cropped_image2=np.zeros((naxes0,naxes1,naxes2))

    Cropped_image1[:,:,:]= np.multiply(im_gnd1[:,:,:], im_gnd3[:,:,:])
    Cropped_image2[:,:,:]= np.multiply(im_gnd2[:,:,:], im_gnd3[:,:,:])

    normalized_cross_correlation= np.sum(np.multiply(Cropped_image1[:,:,:],  Cropped_image2[:,:,:]))/ np.multiply( np.sqrt(np.sum(np.multiply(Cropped_image1[:,:,:],  Cropped_image1[:,:,:]))),  np.sqrt(np.sum(np.multiply(Cropped_image2[:,:,:],  Cropped_image2[:,:,:])))                         )

    return (normalized_cross_correlation)



# convert a text file to matrix
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
#

def Matrix_to_text_file(matrix, text_filename):
    np.savetxt(text_filename, matrix, delimiter='  ')
#

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
# overlap between the two masks

def Bin_dice(rfile,ifile):

    tmp=nib.load(rfile) #segmentation
    d1=tmp.get_data()
    tmp=nib.load(ifile) #segmentation
    d2=tmp.get_data()
    d1_2=np.zeros(np.shape(tmp.get_data()))

    d1_2=len(np.where(d1*d2==1)[0])+0.0
    d1=len(np.where(d1==1)[0])+0.0
    d2=len(np.where(d2==1)[0])+0.0

    if d1==0:
        print ('ERROR: reference image is empty')
        dice=0
    elif d2==0:
        print ('ERROR: input image is empty')
        dice=0
    else:
        dice=2*d1_2/(d1+d2)

    return dice


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


jobs=[]
pool = multiprocessing.Pool(8)


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
    High_resolution_static = args.static # 3D image
    dynamic = args.dyn  # 4D volume

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


    global_mask_basename = 'global_mask'
    global_image_basename = 'flirt_global_static'
    global_matrix_basename = 'global_static_on'
    local_matrix_basename = 'transform_global_on_dyn'
    mask_basename = 'mask_dyn'
    direct_transform_basename='direct_static_on'

##    ##############################################Global registration of the static on each time frame ############################################################################
    for t in range(0, len(dynamicSet)):

        refimage = dynamicSet[t]
        movimage= High_resolution_static
        prefix = dynamicSet[t].split('/')[-1].split('.')[0]
        global_outputimage = outputpath+'flirt_global_static_on_'+prefix+'.nii.gz'
        global_outputmat = outputpath+'global_static_on_'+prefix+'_.mat'

##             ####Global rigid registration of dyn0 on static

        go_init = 'time flirt -noresampblur -searchrx -40 40 -searchry -40 40 -searchrz -40 40  -dof 6 -ref '+refimage+' -in '+movimage+' -out '+global_outputimage+' -omat '+global_outputmat
        jobs.append(go_init)
    pool.map(os.system,jobs)
#    #exit(1)

    global_matrixSet=glob.glob(outputpath+global_matrix_basename+'*.mat')
    global_matrixSet.sort()
##
    global_imageSet=glob.glob(outputpath+global_image_basename+'*.nii.gz')
    global_imageSet.sort()
##
##################Propagate segmentations into the low_resolution domain #################################################################################
    for i in range(0,len(outputpath_boneSet)):
        for t in range(0, len(global_imageSet)):

            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            global_mask= outputpath_boneSet[i]+'/global_mask_'+prefix+'_component_'+str(i)+'.nii.gz'
            go_propagation = 'time flirt -applyxfm -noresampblur -ref '+global_imageSet[t]+' -in ' + args.mask[i] + ' -init '+ global_matrixSet[t] + ' -out ' + global_mask + ' -interp nearestneighbour '
            jobs.append(go_propagation)
    pool.map(os.system,jobs)


#    ####Local Rigid registration of bones########
#
    for i in range(0,len(outputpath_boneSet)):
#
        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()
#
        for t in range(0, len(dynamicSet)):
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            refimage = dynamicSet[t]
            local_outputimage = outputpath_boneSet[i]+'/flirt_global_on_'+prefix+'_component_'+str(i)+'.nii.gz'
            local_outputmat = outputpath_boneSet[i]+'/transform_global_on_'+prefix+'_component_'+str(i)+'.mat'
            #go_init = 'time flirt -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -dof 6 -in '+refimage+' -ref '+global_imageSet[t]+' -out '+local_outputimage+' -omat '+local_outputmat +' -refweight '+ global_maskSet[t]
            go_init = 'time flirt  -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -searchcost normcorr -dof 6 -in '+global_imageSet[t]+' -ref '+refimage+' -out '+local_outputimage+' -omat '+local_outputmat +' -inweight '+ global_maskSet[t]
            jobs.append(go_init)
    pool.map(os.system,jobs)

#### compute local transformations from static to each time frame ################
    for i in range(0,len(outputpath_boneSet)):
#
        local_matrixSet=glob.glob(outputpath_boneSet[i]+'/'+local_matrix_basename+'*.mat')
        local_matrixSet.sort()

        for t in range(0, len(dynamicSet)):

            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            link_matrix=np.dot(Text_file_to_matrix(local_matrixSet[t]), Text_file_to_matrix(global_matrixSet[t]))
            save_path= outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.mat'
            np.savetxt(save_path,link_matrix, delimiter='  ')
######Propagate the high_resolution segmentations into time frames #############################

    for i in range(0,len(outputpath_boneSet)):
        init_matrixSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.mat')
        init_matrixSet.sort()

        for t in range(0, len(dynamicSet)):
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            out2= outputpath_boneSet[i]+'/mask_'+prefix+'_component_'+str(i)+'.nii.gz'
            go_init = 'time flirt -applyxfm -noresampblur -ref '+ dynamicSet[t] + ' -in '+ args.mask[i] + ' -out '+ out2 + ' -init '+init_matrixSet[t]+ ' -interp nearestneighbour '
            jobs.append(go_init)
    pool.map(os.system,jobs)

    ############ Aplly local transformations to the static image  ###########

#    for i in range(0,len(outputpath_boneSet)):
#        init_matrixSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.mat')
#        init_matrixSet.sort()
#        for t in range(0, len(dynamicSet)):
#            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
#            static_on_dyn0= outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.nii.gz'
#            go = 'time flirt -applyxfm -noresampblur -ref '+ dynamicSet[t] + ' -in '+ High_resolution_static + ' -out '+ static_on_dyn0 + ' -init '+init_matrixSet[t]
#            print(go)
#            jobs.append(go)
#    pool.map(os.system,jobs)
#

 ##Fast version:
    for i in range(0,len(outputpath_boneSet)):
        local_matrixSet=glob.glob(outputpath_boneSet[i]+'/'+local_matrix_basename+'*.mat')
        local_matrixSet.sort()
        for t in range(0, len(dynamicSet)):
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            static_on_dyn0= outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.nii.gz'
            go = 'time flirt -applyxfm -ref '+ dynamicSet[t] + ' -in '+ global_imageSet[t] + ' -out '+ static_on_dyn0 + ' -init '+local_matrixSet[t]
            #print(go)
            jobs.append(go)
    pool.map(os.system,jobs)


    ################ finding the time frame that best align with static image  #####################################################################################################################

    dice_evaluation_array=np.zeros((len(outputpath_boneSet), len(dynamicSet)))
    normalized_correlation_evaluation_array=np.zeros((len(outputpath_boneSet), len(dynamicSet)))

    for i in range(0,len(outputpath_boneSet)):

        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()
        maskSet=glob.glob(outputpath_boneSet[i]+'/'+mask_basename+'*.nii.gz')
        maskSet.sort()
        direct_static_on_dynSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.nii.gz')
        direct_static_on_dynSet.sort()


        for t in range(0, len(dynamicSet)):
            normalized_correlation_evaluation_array[i][t] = Normalized_cross_Correlation(direct_static_on_dynSet[t], dynamicSet[t], maskSet[t])
            dice_evaluation_array[i][t] = Bin_dice(maskSet[t],global_maskSet[t])

    linked_time_frame=np.prod(dice_evaluation_array, axis=0)
    t=np.argmax(linked_time_frame)
    if(t<0.5):
        linked_time_frame=np.prod(normalized_correlation_evaluation_array, axis=0)
        t=np.argmax(linked_time_frame)

    ############### Propagate segmentations into the Low-resolution domain ####################################

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
            b=np.dot(inv(Text_file_to_matrix(outputmat)), Text_file_to_matrix(direct_static_on_dynSet[0]))
            Matrix_to_text_file(b, direct_transform)
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
            b=np.dot(inv(Text_file_to_matrix(outputmat)), Text_file_to_matrix(direct_static_on_dynSet[t]))
            Matrix_to_text_file(b, direct_transform)
            direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
            direct_static_on_dynSet.sort()
            out_refweight= output_results+'/mask_'+prefix2+'_component_'+str(i)+'.nii.gz'
            go_propagation = 'time flirt -applyxfm -noresampblur -ref '+dynamicSet[t+1]+' -in ' + args.mask[i] + ' -init '+ direct_static_on_dynSet[t+1] + ' -out ' +out_refweight  + ' -interp nearestneighbour '
            print(go_propagation)
            os.system(go_propagation)
            final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
            final_refweightSet.sort()

            t+=1
