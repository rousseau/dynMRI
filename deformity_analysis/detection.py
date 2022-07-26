#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: CYY
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from sklearn.decomposition import KernelPCA
from scipy.stats import f, zmap, norm, combine_pvalues
from statsmodels.stats.multitest import multipletests
from skimage.morphology import binary_dilation, ball


def data_loading(input_path):
    img = nib.load(input_path)
    img_array = img.get_fdata().astype(float)
    return img_array.flatten()


def img_generation(array, affine, save_path):
    img = nib.Nifti1Image(array, affine)
    nib.save(img, save_path)


def dilation(img_arr):
    img_arr[img_arr != 0] = 1
    kernel = ball(10)
    img_arr_dilated = binary_dilation(img_arr, kernel).astype(float)
    return img_arr_dilated


def detection_t2(temoins, equins, arr_atlas):
    print('Starting Hotelling\'s T2 test')
    n_t = temoins.shape[0]
    n_e = equins.shape[0]
    p_e = np.ones((n_e, shape[0], shape[1], shape[2]), dtype=float)
    p_t = np.ones((n_t, shape[0], shape[1], shape[2]), dtype=float)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if arr_atlas[i, j, k] == 0:
                    pass
                else:
                    mean_t = np.mean(temoins[:, i, j, k], axis=0)
                    cov_t = np.cov(temoins[:, i, j, k], rowvar=False)
                    cov_t_inv = np.linalg.pinv(cov_t)

                    for m in range(n_e):
                        t2 = np.transpose((equins[m, i, j, k] - mean_t)) @ cov_t_inv @ (equins[m, i, j, k] - mean_t)
                        f_stat = (n_t-3)/(3*(n_t-1)) * t2
                        p_e[m, i, j, k] = f.pdf(f_stat, 3, n_t - 3)

                    for n in range(n_t):
                        t2 = np.transpose((temoins[n, i, j, k] - mean_t)) @ cov_t_inv @ (
                                temoins[n, i, j, k] - mean_t)
                        f_stat = (n_t-3)/(3*(n_t-1)) * t2
                        p_t[n, i, j, k] = f.pdf(f_stat, 3, n_t - 3)

    print('Hotelling\'s T2 test finished')
    return p_e


def detection_kpca(temoins, equins, arr_atlas):
    print('Starting KPCA')

    temoins_edge = temoins[:, arr_atlas != 0]
    equins_edge = equins[:, arr_atlas != 0]

    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True).fit(
        temoins_edge.reshape(temoins_edge.shape[0], temoins_edge.shape[1] * 3))

    decomp_t = kpca.transform(temoins_edge.reshape(temoins_edge.shape[0], temoins_edge.shape[1] * 3))
    recons_t = kpca.inverse_transform(decomp_t).reshape(temoins_edge.shape[0], temoins_edge.shape[1], 3)
    p_t = np.power((recons_t - temoins_edge), 2)
    p_t = np.linalg.norm(p_t, axis=2)

    decomp_e = kpca.transform(equins_edge.reshape(equins_edge.shape[0], equins_edge.shape[1] * 3))
    recons_e = kpca.inverse_transform(decomp_e).reshape(equins_edge.shape[0], equins_edge.shape[1], 3)
    p = np.power((recons_e - equins_edge), 2)
    p = np.linalg.norm(p, axis=2)

    z = zmap(scores=p, compare=p_t)
    p_e = norm.pdf(z)
    detection_e = np.ones((equins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2]))
    detection_e[:, arr_atlas != 0] = p_e

    print('KPCA finished')
    return detection_e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Texture plugger')
    parser.add_argument('-b', '--bone', help='input bone image to analysis', type=str, required=True, default='calcaneus.nii.gz')
    parser.add_argument('-t', '--td', help='path to TD subjects\' Warp files', type=str, required=True, default='./td_warp')
    parser.add_argument('-c', '--cp', help='path to CP subjects\' Warp files', type=str, required=True, default='./cp_warp')
    parser.add_argument('-o', '--output', help='path to output files', type=str, required=False, default='./results')

    args = parser.parse_args()

    bone = args.bone
    def_td = args.td
    def_cp = args.cp

    #atlas = './' + bone + '.nii.gz'
    print('loading atlas')
    img_atlas = nib.load(bone)
    affine = img_atlas.affine
    arr_atlas = img_atlas.get_fdata()
    print('atlas dilation')
    arr_atlas = dilation(arr_atlas)

    shape = arr_atlas.shape
    warp_shape = (shape[0], shape[1], shape[2], 3)
    print('warp shape:'+str(warp_shape))

    print('get list of warp files')
    temoins = glob.glob(os.path.join(def_td, '*1Warp.nii.gz'))
    equins = glob.glob(os.path.join(def_cp, '*1Warp.nii.gz'))

    deformation_temoin = np.zeros((len(temoins), shape[0], shape[1], shape[2], 3))
    deformation_equin = np.zeros((len(equins), shape[0], shape[1], shape[2], 3))

    print('loading files')
    for t in range(len(temoins)):
      print(temoins[t])  
      deformation_temoin[t] = np.squeeze(nib.load(temoins[t]).get_fdata().astype(float))  
  
    for e in range(len(equins)):
      print(equins[e])  
      deformation_equin[e] = np.squeeze(nib.load(equins[e]).get_fdata().astype(float))

    """
    deformation_temoin = np.zeros(warp_shape, dtype=float)
    deformation_temoin = deformation_temoin[np.newaxis, :]

    deformation_equin = np.zeros(warp_shape, dtype=float)
    deformation_equin = deformation_equin[np.newaxis, :]

    thr = Pool(processes=5)
    for t in range(len(temoins)):
        print(temoins[t])
        deformation_t = thr.apply_async(data_loading, args=(temoins[t],))
        if t == 0:
            deformation_temoin[0] = deformation_t.get().reshape(shape)
        else:
            deformation_temoin = np.insert(deformation_temoin, t, values=deformation_t.get().reshape(shape), axis=0)

    for e in range(len(equins)):
        deformation_e = thr.apply_async(data_loading, args=(equins[e],))
        if e == 0:
            deformation_equin[0] = deformation_e.get().reshape(shape)
        else:
            deformation_equin = np.insert(deformation_equin, e, values=deformation_e.get().reshape(shape), axis=0)

    thr.close()
    thr.join()
    print('data loaded')
    """

    result_path = args.output
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # score_kpca = np.ones((deformation_equin.shape[0], shape[0], shape[1], shape[2]))
    # score_t2 = np.ones((deformation_equin.shape[0], shape[0], shape[1], shape[2]))
    score = np.ones((deformation_equin.shape[0], shape[0], shape[1], shape[2]))
    confidence = np.zeros((deformation_equin.shape[0], shape[0], shape[1], shape[2]))

    score_kpca = detection_kpca(deformation_temoin, deformation_equin, arr_atlas)
    score_t2 = detection_t2(deformation_temoin, deformation_equin, arr_atlas)

    for n in range(deformation_equin.shape[0]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    _, score[n, i, j, k] = combine_pvalues(np.array([score_kpca[n, i, j, k], score_t2[n, i, j, k]]))
        _, score[n, arr_atlas != 0], _, _ = multipletests(score[n, arr_atlas != 0].flatten())
        confidence[n] = np.ones((shape[0], shape[1], shape[2])) - score[n]

    score_kpca = np.mean(score_kpca, axis=0)
    img = nib.Nifti1Image(score_kpca, affine)
    nib.save(img,
             os.path.join(result_path, bone + '_kpca_mean.nii.gz'))

    score_t2 = np.mean(score_t2, axis=0)
    img = nib.Nifti1Image(score_t2, affine)
    nib.save(img,
             os.path.join(result_path, bone + '_t2_mean.nii.gz'))

    score = np.mean(score, axis=0)
    img = nib.Nifti1Image(score, affine)
    nib.save(img,
             os.path.join(result_path, bone + '_anomap_p.nii.gz'))

    confidence = np.mean(confidence, axis=0)
    img = nib.Nifti1Image(confidence, affine)
    nib.save(img,
             os.path.join(result_path, bone + '_anomap_confidence.nii.gz'))





