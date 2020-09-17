# DeepLearning_on_Ankle_MRI

This software is dedicated to the preparation of MRI data for deep learning for ankle bone segmentation and dynamic sequence reconstruction from a stationnary high-resolution static MRI scan, segmentations of the three ankle bones (calcaneus, talus and tibia) on static MRI scan and a set of low-resolution dynamic MRI scans. 
This project was developped by Patty Coupeau from Polytech Angers during an end-of-study internship at IMT Atlantique.

## Copyright

© IMT Atlantique - LATIM-INSERM UMR 1101

Author(s): Patty Coupeau (patty.coupeau@etud.univ-angers.fr)

## Software requirements

The use of this software need the installation of FSL "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation". Registrations are performed by FLIRT (FMRIB's Linear Image Registration Tool). The decomposition of dynamic sequences into 3D volumes is realized by FSLSPLIT. Required softwares are available ready to run for Mac OS X and Linux (Centos orDebian/Ubuntu) - with Windows computers being supported with a Linux Virtual Machine. 
Deep Learning is performed using the framework PyTorch "https://pytorch.org/".
You also need to install the Python packages Joblib: "https://joblib.readthedocs.io/en/latest/", Nilearn: "https://nilearn.github.io/user_guide.html" and NiBabel: "http://nipy.org/nibabel/installation.html". Data must be in NIFTI format "https://brainder.org/2012/09/23/the-nifti-file-format/".

## Recommendations

In order to use the software, use a directory of type BIDS_dataset composed of two directories:
- sourcedata : with the stationnary high-resolution static MRI scan and the set of low-resolution dynamic MRI scans for all the subjects (each subject in a folder).
- derivatives: with all the data extracted from the sourcedata (segmentations) for all subjects (each subject in a folder)



### extraction footmask on high-resolution static MRI and low-resolution dynamic MRI
##### example: python extraction_footmask.py

This script creates for all subjects in derivatives:
- the foot mask of the HR static MRI in derivatives
- the directory 'volumes' in derivatives composed of:
    - 'volumes3D': 3D volumes resulting from the decomposition of dynamic 4D sequences
    - 'footmask': foot masks of the 3D dynamic volumes



### dilation and blurring of segmentations
##### example: python dilation_blurring_mask.py

This script blurs and dilates the ankle bones segmentations with a radius from 1 to 5 for all subjects in derivatives.
It creates:
- the directory 'segment' in derivatives composed of three sub-directories corresponding to the three bones
- In the directory of each bone, it creates the 5 dilations of segmentations and a directory 'blurred' which contains the 5 segmentations blurred of the bone (corresponding to the 5 radius of dilation).



### registration of static MRI on dynamic volumes
##### example: python Registration_bone_to_bone.py

This script registrates bone-to-bone for all subjects the static MRI on the 3D dynamic volumes and it also registrates the ankle bones segmentations on the dynamic volumes. Only registration with a correlation higher than 0.6 are conserved.
It creates the directory 'registrations' in derivatives containing the registration on all 3D volumes of all dynamic IRM sequences.



### data extraction (patches or slices) for deep learning
##### example: python extract_data_deepLearning.py SegmentationHR patches

2 parameters required:
- problem for which data must be extracted: SegmentationHR (segmentation of static MRI) / Reconstruction / SegmentationLR (segmentation of dynamic MRI)
- type of data: slices / patches

The extraction of slices is implemented only for the problem of SegmentationHR.

To use this function, it is necessary to create on derivatives a directory 'correct_registrations' built as 'registrations' but containing only the visually correct estimated registrations. If you don't have time to analize each result of registration, you can make a copy of 'registrations' and call it 'correct_registrations'.

According to the parameters used, it creates:
- SegmentationHR + patches: the directory DatasetSegmentationHR_patchs with 4 pickle files (patches from static MRI and the three others for the three bones segmentations)
- SegmentationHR + slices: the directory DatasetSegmetnationHR_slices built as for use with patches
- Reconstruction: the directory DatasetReconstruction_patchs composed of three folders corresponding to the three bones. For each bone,there are three pickle files (patches from dynamic volume LR, patches from registrated dynamic volume HR and correlation between both)
- SegmentationLR: the directory DatasetSegmentationLR_patchs composed of three folders corresponding to the three bones. For each bone, there are two pickle files (patches from dynamic volume LR and patches of the registrated bone segmentation)



### Deep Learning for ankle bones segmentation on static MRI
##### example for train: python Train_SegmentationHR.py /mypath/to/TrainData/ /mypath/to/NetworkPATH/ UNet talus

4 positional arguments required :
  - TrainData:    PATH to training data
  - NetworkPATH:  PATH to the network's recording directory
  - Network:      Name of the network to use (UNet/ResNet)
  - Bone:         Name of the bone (calcaneus/talus/tibia)
  
To use this script, the TrainData directory has to be built like a folder for each subject containing the directory DatasetSegmentationHR_patchs created with extract_data_deepLearning.py



##### example for test with patches: python TestSegmentationHR_Patches.py /mypath/to/TestData/ /mypath/to/NetworkPATH/mynetwork.pth /mypath/to/ResultsDirectory/ talus
##### example for test with slices: python TestSegmentationHR_slices.py /mypath/to/TestData/ /mypath/to/NetworkPATH/mynetwork.pth /mypath/to/ResultsDirectory/ talus

4 positional arguments required:
  - TestData:          PATH to testing data
  - NetworkPATH:       PATH to the network to use
  - ResultsDirectory:  PATH to the results storage directory (for graphs and images of some patches)
  - Bone:              Name of the bone (calcaneus/talus/tibia)

To use this script, the TestData directory has to be built like a folder for each subject containing the directory DatasetSegmentationHR_patches (or DatasetSegmentationHR_slices) created with extract_data_deepLearning.py



### Deep Learning for dynamic MRI reconstruction
##### example for train: python TrainReconstruction.py /mypath/to/TrainData/ /mypath/to/NetworkPATH/

2 positional arguments required:
  - TrainData:    PATH to training data
  - NetworkPATH:  PATH to the network recording directory

To use this script, the TrainData directory has to be built like a folder for each subject containing the directory DatasetReconstruction_patches created with extract_data_deepLearning.py


##### example for test: python TestReconstruction.py /mypath/to/TestData/ /mypath/to/NetworkPATH/mynetwork.pth /mypath/to/ResultsDirectory/

3 positional arguments required:
  TestData:          PATH to testing data
  NetworkPATH:       PATH to the network to use
  ResultsDirectory:  PATH to the results storage directory (for graphs and images of some patches)
  
To use this script, the TestData directory has to be built like a folder for each subject containing the directory DatasetReconstruction_patches created with extract_data_deepLearning.py
