# dynMRI

This software is dedicated to the estimation of rigid-body motions and bone deformity analysis for the ankle joint in children from a stationnary high-resolution static MRI scan and a set of low-resolution dynamic MRI scans. It can be also adapted for other articulated structures such as the shoulder etc... 

<!--### Project

* In vivo dynamic evaluation of ankle joint and muscle mechanics in children with spastic equinus deformity due to cerebral palsy: Implications for recurrent equinus.  
* Bone deformity_analysis of children with cerebral palsy using MRI

### Copyright
  © IMT Atlantique - LATIM-INSERM UMR 1101

  Author(s): Karim Makki (karim.makki@imt-atlantique.fr), Yue Cheng (yue.cheng@imt-atlantique.fr), 

  This software is governed by the CeCILL-B license under French law and abiding by the rules of distribution of free software.  You can  use, modify and/ or redistribute the software under the terms of the CeCILL-B license as circulated by CEA, CNRS and INRIA at the following URL "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy, modify and redistribute granted by the license, users are provided only with a limited warranty  and the software's author,  the holder of the economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated with loading,  using,  modifying and/or developing or reproducing the software by the user in light of its specific status of free software, that may mean  that it is complicated to manipulate,  and  that  also therefore means  that it is reserved for developers  and  experienced professionals having in-depth computer knowledge. Users are therefore encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data to be ensured and,  more generally, to use and operate it in the same conditions as regards security.
    The fact that you are presently reading this means that you have had knowledge of the CeCILL-B license and that you accept its terms.
--> 

## In vivo dynamic evaluation of ankle joint and muscle mechanics
Main contributor: Karim Makki (karim.makki@imt-atlantique.fr)
### Requirements
* OS: Mac OS X and Linux
* FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
* NiBabel
* All data to analyse need to be in NIFTI format


### Recommendations

  In order to respect the segmentation protocol, it is highly recommended to segment the tibia for a height of 35 mm. To count from the point of contact between the growth cartilage for the tibia on the one hand and the articular cartilage (tibia / talus) on the other hand.
  This software works well especially when the imaged FOV cover the entire structure of each rigid component (or bone of interest) both in the static and in time frames.
  Please keep your input masks in this order (mask1: calcaneus; mask2: talus; mask3: tibia). 

### Command Line Arguments
##### Example
```
python /motion_estimation/motionEstimation.py  \
  -s <path-to-static-image>  \
  -d <path-to-dynamic-image>   \
  -m <path-to-calcaneus-mask-of-static-image>   \
  -m <path-to-talus-mask-of-static-image>   \
  -m <path-to-tibia-mask-of-static-image>   \
  -o <path-to-save-motion-estimation-output>
```
Please keep your input masks (**-m**) in this order:
`mask1: calcaneus; mask2: talus; mask3: tibia`

### Program outputs
This program outputs a folder per component (or mask):
`/outputpath/propagation/output_path_component_i_/`  
Final results of each component are then saved in a specified sub-folder: 
`/outputpath/propagation/output_path_component_i_/final_results/`   
Final results are:
 - Segmentations of the low-resolution time frames
 - Direct transformations from static to each time frame

### Citation
If this sub-repository helps you, please cite our work using the following BibTex:
```
@article{makki2019vivo,
  title={In vivo ankle joint kinematics from dynamic magnetic resonance imaging using a registration-based framework},
  author={Makki, Karim and Borotikar, Bhushan and Garetier, Marc and Brochard, Sylvain and Salem, Douraied Ben and Rousseau, Fran{\c{c}}ois},
  journal={Journal of biomechanics},
  volume={86},
  pages={193--203},
  year={2019},
  publisher={Elsevier}
}
```

## Bone Deformity Analysis
This is a tool to analyse ankle bone shape difference between typically developping children (TD) and children with cerebral palsy (CP) using 3D MRI.  
Main contributor: Yue Cheng

### Constructed bone shape template
You can fine the TD ankle joint template and the label of bones of interest in `.\deformity_analysis`, named `template_img.nii.gz` and `template_label.nii.gz`  
Also, the shape templates of calcaneus, talus and tibia are provided both in NIFTI and 3D mesh files, named as `bone.nii.gz` or `bone.stl` respectively

### Requirements
* For template estimation and subject-to-template registration: **ANTs** (http://stnava.github.io/ANTs/)
* To install required Python libraries, please use this command:
```
pip install deformity_analysis/requirements.txt
```

### How to Run
To perform the registration, you can run like this:
```
python deformity_analysis/registration.py \ 
  -n <number-of-thread> \
  -t <path-to-TD-subjects-segmentation-files> \
  -c <path-to-CP-subjects-segmentation-files> \
  -ot <path-to-save-TD-subjects-registration-output> \
  -oc <path-to-save-CP-subjects-registration-output>
```

To perform the anomaly analysis, please run like this:
```
python deformity_analysis/detection.py \ 
  -b <which-bone-to-analysis> \
  -t <path-to-TD-subjects-registration-output> \
  -c <path-to-CP-subjects-registration-output> \
  -o <path-to-save-detection-results>
```
Once the NIFTI format detection results are obtained, you can convert the NIFTI to 3D mesh to facilitate the visualization:  
Main contributor of this part: Benjamin Fouquet
```
python deformity_analysis/texture_plugger.py \ 
  -is <path-to-> \
  -t <path-to-TD-subjects-registration-output> \
  -c <path-to-CP-subjects-registration-output> \
  -o <path-to-save-detection-results>
```