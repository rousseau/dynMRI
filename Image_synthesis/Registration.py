import os
import subprocess
import glob
import nibabel as nib
from skimage import morphology
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import sys
import argparse
from os.path import expanduser
home = expanduser("~")

fslcc = '/usr/local/fsl/bin/fslcc'
flirt = '/usr/local/fsl/bin/flirt'

def extract_volumes(args):
    if args.subjects is None:
        subjects = os.listdir(args.derivatives_dir)
    else:
        subjects = args.subjects
    
    for i in range(len(subjects)):
        videos=os.listdir(os.path.join(args.source_dir, subjects[i]))
        for j in range(len(videos)):
            if videos[j].split('_')[2]=='dynamic' and videos[j].split('_')[3]=='MovieClear':
                path = os.path.join(args.result_dir, subjects[i])
                print(path)
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, 'volumes')
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, videos[j][:-7])
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, "volumes3D")
                if not os.path.exists(path):
                    os.mkdir(path)
                
                if len(os.listdir(path)) >= 15:
                    pass
                else:
                    V=nib.load(os.path.join(args.source_dir, subjects[i], videos[j]))
                    affine = V.get_sform()
                    data=V.get_fdata()
                    size=data.shape
                    for k in range(size[3]):
                        volume=data[:,:,:,k]
                        volume=nib.Nifti1Image(volume, affine)
                        nib.save(volume, os.path.join(path, videos[j][:-7]+"_vol"+str(k).rjust(4,'0')+".nii.gz"))
    


def dilation(file_in, file_dilated, r):
    img = nib.load(file_in)
    img_array = img.get_fdata()
    kernel = morphology.ball(r)
    img_array_dilated_bool = morphology.binary_dilation(img_array, kernel)
    img_array_dilated = img_array_dilated_bool.astype(img.get_data_dtype())
    img_dilated = nib.Nifti1Image(img_array_dilated, img.affine)
    nib.save(img_dilated, file_dilated)


def blurring(file_in, file_blurred):
    img = nib.load(file_in)
    img_array = img.get_fdata().astype(float)
    img_array_blurred = gaussian_filter(img_array, 2)
    img_blurred = nib.Nifti1Image(img_array_blurred, img.affine)
    nib.save(img_blurred, file_blurred)


def qualitycontrol(registration,dyn,ind):
    command = "{} {} {}".format(fslcc,registration,dyn)
    ind=0
    corr = subprocess.check_output(command, shell=True)
    corr.replace(b" ",b"")
    corr.replace(b"  ",b"")
    l = corr.split(b'\n')
    if len(l[0])==0:
        print('REGISTERED IMAGE IS EQUAL TO 0')
        return False
    else:
        val = float(l[ind][8:12])
        if (val>=0.62):
            print('\t \t Quality test: OK. Cross correlation = ', str(val))
            return True
        else:
            print('\t \t Quality test: insufficient. Cross correlation = ', str(val))
            return False


def registration(args):
    print('REGISTRATION')

    static_directory = args.source_dir
    segment_directory = args.derivatives_dir
    result_directory = args.result_dir
    dynamic_directory = args.derivatives_dir
        
    if args.subjects is None:
        subjects = os.listdir(segment_directory)
    else:
        subjects = args.subjects

    for i in range(len(subjects)):
        if (subjects[i]!='sub_E11' and subjects[i]!='sub_E12' and subjects[i]!='sub_E04' and subjects[i]!='sub_E07'):# and subjects[i]!='sub_E13' and subjects[i]!='sub_E09'):# and subjects[i]!='sub_T10' and subjects[i]!='sub_T11'):
            print('******************************************************************************')
            print('SUBJECT : '+subjects[i])
        	
            # Get static MRI
            suffixe = '_static_3DT1'
            if os.path.exists(os.path.join(static_directory, subjects[i],subjects[i]+suffixe+'_flipCorrected.nii.gz')):
                file_in = os.path.join(static_directory, subjects[i],subjects[i]+suffixe+'_flipCorrected.nii.gz')
            else:
                file_in = os.path.join(static_directory, subjects[i], subjects[i]+suffixe + '.nii.gz')
        	
            # Get static segmentations
            suffixe = subjects[i] + '_static_3DT1_segment_calcaneus'
            if os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_flipped_binarized.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, subjects[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, subjects[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_flipCorrected.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, subjects[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, subjects[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe + '.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, subjects[i], suffixe + '.nii.gz')
            else:
                file_segment_calcaneus = os.path.join(segment_directory, subjects[i], subjects[i]+'_static_3DT1_segment_smooth_calcaneus.nii.gz')
        
            suffixe = subjects[i] + '_static_3DT1_segment_talus'
            if os.path.exists(os.path.join(segment_directory, subjects[i],suffixe+'_flipped_binarized.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, subjects[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, subjects[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_flipCorrected.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, subjects[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, subjects[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe + '.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, subjects[i], suffixe + '.nii.gz')
            else:
                file_segment_talus = os.path.join(segment_directory, subjects[i], subjects[i]+'_static_3DT1_segment_smooth_talus.nii.gz')
        
            suffixe = subjects[i] + '_static_3DT1_segment_tibia'
            if os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_flipped_binarized.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, subjects[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, subjects[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i],suffixe+'_flipCorrected.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, subjects[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, subjects[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe + '.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, subjects[i],suffixe + '.nii.gz')
            else:
                file_segment_tibia = os.path.join(segment_directory, subjects[i], subjects[i]+'_static_3DT1_segment_smooth_tibia.nii.gz')
        	
            # Create the registration results directory
            if not os.path.exists(os.path.join(result_directory,subjects[i])):
                os.mkdir(os.path.join(result_directory,subjects[i]))
        
            
            images=os.listdir(os.path.join(dynamic_directory,subjects[i], 'volumes'))

            for j in range(len(images)):
                # Get dynamic MRI "MovieClear"
                if images[j].find('MovieClear')!=-1 and not(subjects[i]=='sub_E03' and images[j].find('10')!=-1):
                    if not(subjects[i]=='sub_T01' and images[j].find('flipCorrected')==-1):
                        print('*Vidéo : '+str(images[j]))
        
                        # Create subdirectories for registrations
                        if not os.path.exists(os.path.join(result_directory,subjects[i], 'registrations')):
                            os.mkdir(os.path.join(result_directory,subjects[i], 'registrations'))
                        if not os.path.exists(os.path.join(result_directory,subjects[i], 'registrations', images[j].replace('.nii.gz',''))):
                            os.mkdir(os.path.join(result_directory,subjects[i], 'registrations',images[j].replace('.nii.gz','')))
                
                        # Registration of static MRI on dynamic MRI
                        volumes = os.listdir(os.path.join(dynamic_directory,subjects[i], 'volumes', images[j].replace('.nii.gz',''), 'volumes3D'))

                        for k in range(len(volumes)):
                            print('\t *Volume : '+volumes[k])
                            file_ref = os.path.join(dynamic_directory,subjects[i], 'volumes',images[j].replace('.nii.gz',''), 'volumes3D', volumes[k])
                            recording_directory = os.path.join(result_directory,subjects[i], 'registrations', images[j].replace('.nii.gz',''))

                            if os.path.exists(os.path.join(result_directory,subjects[i], 'registrations',images[j],volumes[k].replace('.nii.gz','')+'_registration.nii.gz')):
                                pass
                            else:
                                print('\t \t REGISTRATION.......')
                                command = "{} -in {} -ref {} -out {} -omat {} -dof 6".format(flirt,
                                        file_in,
                                        file_ref,
                                        os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.nii.gz'),
                                        os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                os.system(command)
        
                            # Registration quality control
                            registration = os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.nii.gz')
                            if(qualitycontrol(registration,file_ref,k)):
                                for bone in ['calcaneus', 'talus', 'tibia']:

                                    # Registration bone-to-bone
                                    masks = os.listdir(os.path.join(segment_directory, subjects[i], 'segment', bone+'_dilated', 'blurred'))
                                    for blurred in masks:
                                        if bone == 'calcaneus':
                                            if 'r1' in blurred:
                                                mask = blurred
                                        elif bone == 'talus' or bone == 'tibia':
                                            if 'r2' in blurred:
                                                mask = blurred
                    
                                    if os.path.exists(os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz')):
                                        pass
                                    else:
                                        print('\t \t ' + volumes[k] + ' - ' + bone)
                                        command = '{} -in {} -ref {} -inweight {} -out {} -omat {} -init {} -dof 6 -nosearch'.format(flirt,
                                                file_in,
                                                file_ref,
                                                os.path.join(segment_directory, subjects[i], 'segment', bone+'_dilated', 'blurred',mask),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz'),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                        os.system(command)
                                        
                                    # Registration static segmentation
                                    if os.path.exists(os.path.join(result_directory,subjects[i],images[j],volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz')):
                                        pass
                                    else:
                                        if bone == 'calcaneus':
                                            file_segment = file_segment_calcaneus
                                        elif bone == 'talus':
                                            file_segment = file_segment_talus
                                        elif bone == 'tibia':
                                            file_segment = file_segment_tibia
                                        command = '{} -in {} -ref {} -out {} -init {} -applyxfm'.format(flirt,
                                                file_segment,
                                                file_ref,
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz'),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'))
                                        os.system(command)

        				



def blur_static(args):
    print('DILATATION + BLURRING')

    recording_path=args.result_dir
    segment_directory = args.derivatives_dir
    number_os=0
    
    while(number_os<3):
        if(number_os==0):
            bone = 'calcaneus'
        elif(number_os==1):
            bone='talus'
        elif(number_os==2):
            bone='tibia'

        if args.subjects is None:
            subjects = os.listdir(segment_directory)
        else:
            subjects = args.subjects
    
        for i in range(len(subjects)):
            print(subjects[i])

            suffixe = subjects[i] + '_static_3DT1_segment_' + bone
            if os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_flipped_binarized.nii.gz')):
                seg = os.path.join(segment_directory, subjects[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                seg = os.path.join(segment_directory, subjects[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_flipCorrected.nii.gz')):
                seg = os.path.join(segment_directory, subjects[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe+'_binarized.nii.gz')):
                seg = os.path.join(segment_directory, subjects[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, subjects[i], suffixe + '.nii.gz')):
                seg = os.path.join(segment_directory, subjects[i], suffixe + '.nii.gz')
            else:
                seg = os.path.join(segment_directory, subjects[i], subjects[i]+'_static_3DT1_segment_smooth_calcaneus.nii.gz')


            file_dilated_path = os.path.join(recording_path, subjects[i])
            if not os.path.isdir(file_dilated_path):
                os.mkdir(file_dilated_path)
            file_dilated_path = os.path.join(file_dilated_path, 'segment')
            if not os.path.isdir(file_dilated_path):
                os.mkdir(file_dilated_path)
            file_dilated_path = os.path.join(file_dilated_path, bone + '_dilated')
            if not os.path.isdir(file_dilated_path):
                os.mkdir(file_dilated_path)
            
            radius_min = 1
            radius_max = 5
            
            file_blurred_path = os.path.join(file_dilated_path, 'blurred')
            if not os.path.isdir(file_blurred_path):
                os.mkdir(file_blurred_path)

            
        
            if os.path.exists(seg):
                print(seg)

                for r in range(radius_min, radius_max+1):
                    prefix = suffixe + '_static_dilated_r' + str(r)               
                    
                    file_dilated = os.path.join(file_dilated_path, prefix + '.nii.gz')
                    if os.path.exists(file_dilated):
                        pass
                    else:
                        dilation(seg, file_dilated, r)
                    
                    #Floutage des masques
                    file_blurred = os.path.join(file_blurred_path, prefix + '_blurred.nii.gz')
                    if os.path.exists(file_blurred):
                        pass
                    else:
                        blurring(file_dilated, file_blurred)
                        print(file_blurred)
            else:
                print('File '+suffixe+'.nii.gz not found')    
        number_os+=1



if __name__ == '__main__':
    '''
        Required:
            - "sourcedata" directory: contains all sourcedata per subject
            - "derivatives" directory: contains ankle joint bones segmentations for dynamic MRI
    '''

    parser = argparse.ArgumentParser(description='HR image degradation')
    parser.add_argument('--source_dir', help='Path to static directory', type=str, required=False, default=os.path.join(home, 'Equinus_BIDS_dataset', 'sourcedata'))
    parser.add_argument('--derivatives_dir', help='Path to derivative directory', type=str, required=False, default=os.path.join(home, "Equinus_BIDS_dataset","derivatives"))
    parser.add_argument('--result_dir', help="Path to save directory", type=str, required=False, default=os.path.join(home, "Equinus_BIDS_dataset","derivatives"))
    parser.add_argument('--subjects', help='Subject(s) to register (ex: sub_E01 sub_T10). Default = all accessible subjects', nargs='+', type=str, required=False, default=None)
    args = parser.parse_args()

    print('*** ARGUMENTS ***')
    print('{:>25}: {}'.format("Static directory", args.source_dir))
    print('{:>25}: {}'.format("Derivatives directory", args.derivatives_dir))
    print('{:>25}: {}'.format("Result directory", args.result_dir))
    print('{:>25}: {}{}\n'.format("Subjects", str(args.subjects), " (if None: use all accessible subjects)"))
    print('*****************')

    blur_static(args)
    extract_volumes(args)
    registration(args)
