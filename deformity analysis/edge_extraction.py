import os
import glob
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool


def generation(sujet, seg_arr, data_path, affine):
    num = sujet.rsplit('/', 1)[1].split('_')[1]
    print(num)
    img = nib.load(sujet)
    img_arr = np.squeeze(img.get_fdata())
    arr = np.stack((img_arr[:, :, :, 0] * seg_arr, img_arr[:, :, :, 1] * seg_arr, img_arr[:, :, :, 2] * seg_arr), axis=3)
    img_edge = nib.Nifti1Image(arr, affine)
    nib.save(img_edge, os.path.join(data_path, num + '_edge.nii.gz'))


def generation_template_edge(seg, data_save_path):
    img_seg = sitk.ReadImage(seg)
    img_seg_float = sitk.Cast(img_seg, sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_seg = sobel_op.Execute(img_seg_float)
    sitk.WriteImage(sobel_seg, data_save_path)

    img_seg = nib.load(data_save_path)
    seg_arr = img_seg.get_fdata().astype(float)
    print(np.max(seg_arr))

    seg_arr[seg_arr < 3] = 0
    seg_arr[seg_arr >= 3] = 1.0

    affine = img_seg.affine
    img_seg_new = nib.Nifti1Image(seg_arr, affine)
    nib.save(img_seg_new, data_save_path)
    return seg_arr, affine


if __name__ == '__main__':
    bones = ['calcaneus', 'talus', 'tibia']
    seg_arr = np.zeros((323, 323, 144))
    for b in bones:
        save_path = './' + b + '/deformation_edge'
        seg_atlas = './atlas_unet3_' + b + '.nii.gz'
        edge_seg_atlas_save_path = './atlas_unet3_' + b + '.nii.gz'
        seg_arr, affine = generation_template_edge(seg_atlas, edge_seg_atlas_save_path)

        sujets = glob.glob('./warp/*.nii.gz')
        thr = Pool(processes=6)
        for i in range(len(sujets)):
            thr.apply_async(generation, args=(sujets[i], seg_arr, save_path, affine,))
        thr.close()
        thr.join()






    







