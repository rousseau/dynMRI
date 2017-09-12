# dynMRI

This software is dedicated to the estimation of rigid-body motions for the ankle joint in children from a stationnary high-resolution static MRI scan and a set of low-resolution dynamic MRI scans. It can be also adapted for other articulated structures such as the shoulder etc... 

### Project

In vivo dynamic evaluation of ankle joint and muscle mechanics in children with spastic equinus deformity due to cerebral palsy: Implications for recurrent equinus.

### Copyright
  © IMT Atlantique - LATIM-INSERM UMR 1101

  Author(s): Karim Makki (karim.makki@imt-atlantique.fr)

  This software is governed by the CeCILL-B license under French law and abiding by the rules of distribution of free software.  You can  use, modify and/ or redistribute the software under the terms of the CeCILL-B license as circulated by CEA, CNRS and INRIA at the following URL "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy, modify and redistribute granted by the license, users are provided only with a limited warranty  and the software's author,  the holder of the economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated with loading,  using,  modifying and/or developing or reproducing the software by the user in light of its specific status of free software, that may mean  that it is complicated to manipulate,  and  that  also therefore means  that it is reserved for developers  and  experienced professionals having in-depth computer knowledge. Users are therefore encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data to be ensured and,  more generally, to use and operate it in the same conditions as regards security.
    The fact that you are presently reading this means that you have had knowledge of the CeCILL-B license and that you accept its terms.
    
### Software requirements

The use of this software need the installation of FSL "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation".
Registrations are performed by FLIRT (FMRIB's Linear Image Registration Tool). Required softwares are available ready to run for Mac OS X and Linux (Centos orDebian/Ubuntu) - with Windows computers being supported with a Linux Virtual Machine.
You also need to install the Python package NiBabel: "http://nipy.org/nibabel/installation.html".
Data must be in NIFTI format "https://brainder.org/2012/09/23/the-nifti-file-format/". 

### Recommendations

  Inorder to respect the segmentation protocol, it is highly recommended to segment the tibia for a height of 35 mm. To count from the point of contact between the growth cartilage for the tibia on the one hand and the articular cartilage (tibia / talus) on the other hand.
  This software works well especially when the imaged FOV cover the entire structure of each rigid component (or bone of interest) both in the static and in time frames.
  Please keep your input masks in this order (mask1: calcaneus; mask2: talus; mask3: tibia). 

### Command Line Arguments
##### Example

python motionEstimation.py  -s /home/karim/Data/subject07/static07.nii.gz  -d /home/karim/Data/subject07/dynamic07.nii.gz   -m /home/karim/Data/subject07/segment/mask1.nii.gz   -m /home/karim/Data/subject07/segment/mask2.nii.gz   -m  /home/karim/Data/subject07/segment/mask3.nii.gz   -o /home/karim/Exp/septembre/jeudi/subject07/dynamic07/

-s | -d | -m | -o
  (required) 

    -s - filename of the 3D static image
    -d - filename of the 4D dynamic sequence 
    -m - mask filename : Accepted multiple times
    -o - Output path

### Program outputs
This program outputs a folder per component (or mask): (/outputpath/propagation/output_path_component_i_/). Final results of each component are then saved in a specified sub-folder: (/outputpath/propagation/output_path_component_i_/final_results/). 
Final results are:
 - segmentations of the low-resolution time frames
 - Direct transformations from static to each time frame

