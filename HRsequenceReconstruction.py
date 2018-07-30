
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
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--floating', help='Floating high-resolution image', type=str, required = True)
    parser.add_argument('-HRsegments', '--HRsegments', help='High_resolution segments path', type=str, required = True)
    parser.add_argument('-datapath', '--estimated', help='Transformation path, containing N folders where N is the total number of components, each folder contains T estimated transformations ', type=str, required = True)
    parser.add_argument('-transformFusion', '--transformFusion', help='transformFusion.py complete path', type=str, required = True)
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)
    parser.add_argument('-tbasename', '--tbasename', help='Transformation basename', type=str, required = True)
    parser.add_argument('-os', '--OperatingSystem', help='Operating system: 0 if Linux and 1 if Mac Os', type=int, required = True)

    args = parser.parse_args()



    if (int(args.OperatingSystem) == 0):
        call_flirt = 'fsl5.0-flirt'
        call_fslmath = 'fsl5.0-fslmaths'

    elif (int(args.OperatingSystem) == 1):
        call_flirt = 'flirt'
        call_fslmath = 'fslmaths'

    else :
        print(" \n Please select your Operating System: 0 if Linux and 1 if Mac Os \n")

    HRsegmentSet=glob.glob(args.HRsegments+'*nii.gz')
    HRsegmentSet.sort()

    go = call_flirt +' -usesqform -applyxfm -noresampblur -ref '+args.floating   #' -init '+ global_matrixSet[t] + ' -out ' + global_mask + ' -interp nearestneighbour '


######### Compute warped high_resolution segmentations #######################

    for i in range (0, len(HRsegmentSet)):
        HR_component_path=args.output+'High_resolution_component'+str(i)
        if not os.path.exists(HR_component_path):
           os.makedirs(HR_component_path)

        transformSet=glob.glob(args.estimated+'output_path_component'+str(i)+'/final_results/'+'*.mat')
        transformSet.sort()


        LR_componentSet=glob.glob(args.estimated+'output_path_component'+str(i)+'/final_results/'+'*.nii.gz')
        LR_componentSet.sort()


        go2 = go + ' -in ' + HRsegmentSet[i]


        for t in range(0, len(transformSet)):

            prefix = LR_componentSet[t].split('/')[-1].split('.')[0]


            go3 = go2 + ' -init '+ transformSet[t] + ' -out '+ HR_component_path +'/HR_'+prefix+'.nii.gz'

            print(go3)
            os.system(go3)

            binarization = call_fslmath + ' ' + HR_component_path +'/HR_'+prefix+'.nii.gz' + ' -thr 0.9  -bin '+  HR_component_path +'/HR_'+prefix+'.nii.gz'
            os.system(binarization)


######## Hr sequence reconstruction ###########################


#### Directory in which all reconstructed time frames will be saved

    reconstruction_directory = args.output+'Reconstructed_sequence/'
    if not os.path.exists(reconstruction_directory):
           os.makedirs(reconstruction_directory)


    for t in range(0, len(LR_componentSet)):

        reconstruction = ' python ' + args.transformFusion + ' -in ' + args.floating

        for i in range (0, len (HRsegmentSet)):

            HR_componentSet=glob.glob(args.output+'High_resolution_component'+str(i)+'/HR_mask_'+'*.nii.gz')
            HR_componentSet.sort()

            transformSet=glob.glob(args.estimated+'output_path_component'+str(i)+'/final_results/'+'*.mat')
            transformSet.sort()

            reconstruction+= ' -refweight ' + HR_componentSet[t] + ' -t ' + transformSet[t]


        reconstruction += ' -o ' + reconstruction_directory +   ' -warped_image  High_resolution_reconstructed_dyn'+str(t)+'.nii.gz' + ' -def_field  Deformation_field_dyn'+str(t)+'.nii.gz'
        print(reconstruction)
        os.system(reconstruction)
