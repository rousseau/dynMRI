import os
import os.path
import glob
import multiprocessing
import nibabel as nib
from xlwt import Workbook
import argparse
import numpy as np

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_path', help='Data path', type=str, required = True)
    parser.add_argument('-motionEstimation', '--script_path', help='motionEstimation.py path', type=str, required = True)
    parser.add_argument('-e', '--excel_file', help='Output Excel file', type=str, required = True)
    parser.add_argument('-o', '--output_path', help='Output directory', type=str, required = True)
    parser.add_argument('-os', '--OperatingSystem', help='Operating system: 0 if Linux and 1 if Mac Os', type=int, required = True)
    args = parser.parse_args()

    jobs=[]
    pool = multiprocessing.Pool(8)


    data_path   = args.data_path #'/home/karim/Data/Francois/'
    output_path = args.output_path #'/home/karim/Exp/septembre/tuesday'
    save_path   = output_path+'/'+args.excel_file #excel_file.xls'


    subject_basename='subject'
    data_basename='201'
    mask_basename='smoothed_segment'

    subjectSet=glob.glob(data_path+subject_basename+'*')
    subjectSet.sort()


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

            sheet1.write(subject+1,component+1,Bin_dice(ground_truth_time_frameSet[0],time_frameSet[0]))
            sheet2.write(subject+1,component+1,Bin_dice(ground_truth_time_frameSet[1],time_frameSet[len(time_frameSet)-1]))


book.save(save_path)
