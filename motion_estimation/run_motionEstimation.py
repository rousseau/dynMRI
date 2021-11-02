# -*- coding: utf-8 -*-

import os
import os.path
import glob
import multiprocessing
import nibabel as nib
from xlwt import Workbook
import argparse
import numpy as np
import math
import medpy
from medpy.metric.binary import hd


def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())

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

## calculate Hausdorff distance between 3D grids directly ######

### grabbing a box around a given point and taking all the radius inorder to reduce the number of points required to check.
def bbox(array, point, radius):
    a = array[np.where(np.logical_and(array[:, 0] >= point[0] - radius, array[:, 0] <= point[0] + radius))]
    b = a[np.where(np.logical_and(a[:, 1] >= point[1] - radius, a[:, 1] <= point[1] + radius))]
    c = b[np.where(np.logical_and(b[:, 2] >= point[2] - radius, b[:, 2] <= point[2] + radius))]
    return c


###  hausdroff distance calculation

def hausdorff(surface_a, surface_b):

    # Taking two arrays as input file, the function is searching for the Hausdorff distane of "surface_a" to "surface_b"
    dists = []

    l = len(surface_a)

    for i in xrange(l):

        # walking through all the points of surface_a
        dist_min = 1000.0
        radius = 0
        b_mod = np.empty(shape=(0, 0, 0))

        # increasing the cube size around the point until the cube contains at least 1 point
        while b_mod.shape[0] == 0:
            b_mod = bbox(surface_b, surface_a[i], radius)
            radius += 1

        # to avoid getting false result (point is close to the edge, but along an axis another one is closer),
        # increasing the size of the cube
        b_mod = bbox(surface_b, surface_a[i], radius * math.sqrt(3))

        for j in range(len(b_mod)):
            # walking through the small number of points to find the minimum distance
            dist = np.linalg.norm(surface_a[i] - b_mod[j])
            if dist_min > dist:
                dist_min = dist

        dists.append(dist_min)

    return np.max(dists)

def Rotation_vector_from_flirt_transform(matrix):  ### equivalent to the fsl tool, avscale

        rotation_vector=np.zeros(3)
        rotation_vector[1]= -math.asin(matrix[0,2])
        c5= math.cos(rotation_vector[1])
        rotation_vector[1]= 180*rotation_vector[1]/math.pi
        rotation_vector[0]= 180*math.atan2((matrix[1,2] / c5),(matrix[2,2] / c5))/math.pi
        rotation_vector[2]= 180*math.atan2((matrix[0,1] / c5),(matrix[0,0] / c5))/math.pi
        np.set_printoptions(precision=6, suppress=True)

        return rotation_vector #return rotation vector in degrees [Rx Ry Rz]


def Translation_vector_from_flirt_transform(matrix):    ### equivalent to the fsl tool, avscale

    return [matrix[0,3], matrix[1,3], matrix[2,3]] #return translation vector in mm [Tx Ty Tz]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_path', help='Data path', type=str, required = True)
    parser.add_argument('-motionEstimation', '--script_path', help='motionEstimation.py path', type=str, required = True)
    parser.add_argument('-e', '--excel_file', help='Output Excel file', type=str, required = True)
    parser.add_argument('-o', '--output_path', help='Output directory', type=str, required = True)
    parser.add_argument('-os', '--OperatingSystem', help='Operating system: 0 if Linux and 1 if Mac Os', type=int, required = True)
    args = parser.parse_args()

    jobs=[]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())


    data_path   = args.data_path #'/home/karim/Data/Francois/'
    output_path = args.output_path #'/home/karim/Exp/septembre/tuesday'
    save_path   = output_path+'/'+args.excel_file #excel_file.xls'

    print(args.data_path)


    subject_basename='subject'
    data_basename='201'
    mask_basename='smoothed_segment'

    subjectSet = glob.glob(data_path+subject_basename+'*')
    subjectSet.sort()

    print(subjectSet)


    ### Excel file creation

    book = Workbook()

    sheet1 = book.add_sheet('time_frame0')
    sheet2 = book.add_sheet('time_frameN')

    sheet1.write(0,1,'calcaneus')
    sheet1.write(0,2,'talus')
    sheet1.write(0,3,'tibia')

    sheet2.write(0,1,'calcaneus')
    sheet2.write(0,2,'talus')
    sheet2.write(0,3,'tibia')


    for subject in range (0, len(subjectSet)):

        static_image=glob.glob(subjectSet[subject]+'/'+data_basename+'*.nii.gz')
        go='python ' + args.script_path +' -s '+ static_image[0] + ' -os '+ str(args.OperatingSystem)

        subject_name=os.path.basename(subjectSet[subject])
        output_path2= output_path + '/'+subject_name

        sheet1.write(subject+1,0,subject_name)
        sheet2.write(subject+1,0,subject_name)

        maskSet=glob.glob(subjectSet[subject]+'/segment/smoothed_segment/'+mask_basename+'*.nii.gz')
        maskSet.sort()

        go1= go+ ' -m '+ maskSet[0] + ' -m '+ maskSet[1] + ' -m '+ maskSet[2]

        dynamic_sequenceSet=glob.glob(subjectSet[subject]+'/dynamic/'+data_basename+'*')
        dynamic_sequenceSet.sort()

        sequence_name=os.path.basename(dynamic_sequenceSet[0])
        sequence_name_without_extension = sequence_name[:sequence_name.index('.')]

        output_path3= output_path2+'/'+sequence_name_without_extension+'/'
        if not os.path.exists(output_path3):
            os.makedirs(output_path3)

        go2= go1 + ' -d ' + dynamic_sequenceSet[0] + ' -o ' + output_path3
        print(go2)
        jobs.append(go2)
    pool.map(os.system,jobs)

    pool.close()
    pool.join()
    ################# Compute Dice score between automated segmentations and ground truth ##########################


    for subject in range (0, len(subjectSet)):

        subject_name=os.path.basename(subjectSet[subject])

        dynamic_sequenceSet=glob.glob(output_path+'/'+subject_name+'/*')
        dynamic_sequenceSet.sort()

        ground_truth_sequenceSet=glob.glob(data_path+'ground_truth/'+subject_name+'/*')
        ground_truth_sequenceSet.sort()

        componentSet=glob.glob(dynamic_sequenceSet[0]+'/propagation/*')
        componentSet.sort()

        ground_truth_componentSet=glob.glob(ground_truth_sequenceSet[0]+'/*')
        ground_truth_componentSet.sort()

        for component in range(0, len(componentSet)):
            time_frameSet=glob.glob(componentSet[component]+'/final_results/'+'*.nii.gz')
            time_frameSet.sort()

            ground_truth_time_frameSet=glob.glob(ground_truth_componentSet[component]+'/'+'*.nii.gz')
            ground_truth_time_frameSet.sort()
##### Dice score metric

            #sheet1.write(subject+1,component+1,Bin_dice(ground_truth_time_frameSet[0],time_frameSet[0]))
            #sheet2.write(subject+1,component+1,Bin_dice(ground_truth_time_frameSet[1],time_frameSet[len(time_frameSet)-1]))

##### Hausdroff distance metric
            im1 = nifti_to_array(ground_truth_time_frameSet[0])
            im2 = nifti_to_array(time_frameSet[0])
            im3 = nifti_to_array(ground_truth_time_frameSet[1])
            im4 = nifti_to_array(time_frameSet[len(time_frameSet)-1])


            #sheet1.write(subject+1,component+1,hausdorff(np.argwhere(im1!=0), np.argwhere(im2!=0)))
            #sheet2.write(subject+1,component+1,hausdorff(np.argwhere(im3!=0), np.argwhere(im4!=0)))

            sheet1.write(subject+1,component+1, hd(im1, im2))
            sheet2.write(subject+1,component+1, hd(im3, im4))




book.save(save_path)
