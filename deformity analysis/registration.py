import os


def antsRegistration(fixedImage, fixedMask, fixedDist, movingImage, movingMask, movingDist, outputPrefix_image):
    registration = 'antsRegistration -d 3 ' + \
                   '-m CC[' + fixedImage + ',' + movingImage + ',1,4] ' + \
                   '-m MeanSquares[' + fixedMask + ',' + movingMask + ',1,0] ' + \
                   '-m MeanSquares[' + fixedDist + ',' + movingDist + ',1,0] ' + \
                   '-t Rigid[0.1] -f 6x4x2x1 -s 3x2x1x0 -c 100x100x70x20 ' + \
                   '-m CC[' + fixedImage + ',' + movingImage + ',1,4] ' + \
                   '-m MeanSquares[' + fixedMask + ',' + movingMask + ',1,0] ' + \
                   '-m MeanSquares[' + fixedDist + ',' + movingDist + ',1,0] ' + \
                   '-t Affine[0.1] -f 6x4x2x1 -s 3x2x1x0 -c 100x100x70x20 ' + \
                   '-m CC[' + fixedImage + ',' + movingImage + ',1,4] ' + \
                   '-m MeanSquares[' + fixedMask + ',' + movingMask + ',1,0] ' + \
                   '-m MeanSquares[' + fixedDist + ',' + movingDist + ',1,0] ' + \
                   '-t SyN[0.1] -f 6x4x2x1 -s 3x2x1x0 -c 100x100x70x20 -v 1 ' + \
                   '-o [' + outputPrefix_image + ',' + outputPrefix_image + 'Warped.nii.gz,' + outputPrefix_image + 'InverseWarped.nii.gz]'

    print(registration)
    os.system(registration)
    registration_mask = 'antsApplyTransforms -d 3 -i ' + movingMask + ' -r ' + fixedMask + ' -o ' + outputPrefix_mask \
                        + ' -t ' + outputPrefix_image + '1Warp.nii.gz -t ' + outputPrefix_image + '0GenericAffine.mat'
    os.system(registration_mask)


if __name__ == '__main__':
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '12'
    dataDirectory = './exp_reso05_seg_unet'

    fixedImage = './atlas_unet3template0.nii.gz'
    fixedMask = './atlas_unet3template1.nii.gz'
    fixedDist = './atlas_unet3template2.nii.gz'

    moving_type = 'subjects'  # 'subjects' or 'atlas'

    if moving_type == 'atlas':
        movingImage = './atlas_equin_unet3template0.nii.gz'
        movingMask = './atlas_equin_unet3template1.nii.gz'
        movingDist = './atlas_equin_unet3template2.nii.gz'
        outputPrefix_image = './registration_atlas_to_atlas/atlas_equins_'
        antsRegistration(fixedImage, fixedMask, fixedDist, movingImage, movingMask, movingDist, outputPrefix_image)

    else:
        outputDirectory = './registration_equins_to_atlas'
        movs = ['E01', 'E02', 'E03', 'E05', 'E06', 'E08', 'E09', 'E10', 'E11', 'E13']
        for i in range(len(movs)):
            print(movs[i])
            movingImage = os.path.join(dataDirectory, 'sub_' + movs[i] + '_static_3DT1_flirt.nii.gz')
            movingMask = os.path.join(dataDirectory, 'sub_' + movs[i] + '_static_3DT1_flirt_fuzzy.nii.gz')
            movingDist = os.path.join(dataDirectory, 'sub_' + movs[i] + '_static_3DT1_flirt_dist.nii.gz')
            outputPrefix_image = os.path.join(outputDirectory, 'sub_' + movs[i] + '_on_atlas_')
            outputPrefix_mask = os.path.join(outputDirectory, 'sub_' + movs[i] + '_on_atlas_segment.nii.gz')
            antsRegistration(fixedImage, fixedMask, fixedDist, movingImage, movingMask, movingDist, outputPrefix_image)





