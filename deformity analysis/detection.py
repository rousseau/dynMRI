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
from scipy import stats
from multiprocessing import Pool
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from pyanom.outlier_detection import HotelingT2
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
    img_arr_dilated = binary_dilation(img_arr, kernel).astype(img.get_data_dtype())
    return img_arr_dilated


def detection_t2(temoins, equins, arr_atlas):
    p = np.zeros((equins.shape[0], 323, 323, 144, 3), dtype=float)
    p_t = np.zeros((temoins.shape[0], 323, 323, 144, 3), dtype=float)
    print('Starting Hotelling\'s T2 test')

    for i in range(323):
        for j in range(323):
            for k in range(144):
                if arr_atlas[i, j, k] == 0:
                    pass
                else:
                    for m in range(3):
                        model = HotelingT2()
                        model.fit((temoins[:, i, j, k, m].reshape(-1, 1)))
                        temoin_score = model.score(temoins[:, i, j, k, m].reshape(-1, 1))
                        p_t[:, i, j, k, m] = temoin_score.reshape(temoins.shape[0], )
                        anomaly_score = model.score(equins[:, i, j, k, m].reshape(-1, 1))
                        p[:, i, j, k, m] = anomaly_score.reshape(equins.shape[0], )
    p = np.mean(p, axis=4)
    p_t = np.mean(p_t, axis=4)

    print('Hotelling\'s T2 test finished')
    return p, p_t


def detection_pca(temoins, equins, arr_atlas):
    print('Starting PCA')

    temoins_edge = temoins[:, arr_atlas != 0]
    equins_edge = equins[:, arr_atlas != 0]

    pca = PCA().fit(temoins_edge.reshape(temoins_edge.shape[0], temoins_edge.shape[1] * 3))

    decomp_t = pca.transform(temoins_edge.reshape(temoins_edge.shape[0], temoins_edge.shape[1] * 3))
    recons_t = pca.inverse_transform(decomp_t).reshape(temoins_edge.shape[0], temoins_edge.shape[1], 3)
    p_t = np.power((recons_t - temoins_edge), 2)
    p_t = np.mean(p_t, axis=2)
    reconstruction_t = np.zeros((temoins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2], 3))
    reconstruction_t[:, arr_atlas != 0] = recons_t
    detection_t = np.zeros((temoins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2]))
    detection_t[:, arr_atlas != 0] = p_t

    decomp_e = pca.transform(equins_edge.reshape(equins_edge.shape[0], equins_edge.shape[1] * 3))
    recons_e = pca.inverse_transform(decomp_e).reshape(equins_edge.shape[0], equins_edge.shape[1], 3)
    p = np.power((recons_e - equins_edge), 2)
    p = np.mean(p, axis=2)
    reconstruction_e = np.zeros((equins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2], 3))
    reconstruction_e[:, arr_atlas != 0] = recons_e
    detection_e = np.zeros((equins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2]))
    detection_e[:, arr_atlas != 0] = p

    print('PCA finished')
    return detection_e, detection_t


def detection_kpca(temoins, equins, arr_atlas):
    print('Starting KPCA')

    temoins_edge = temoins[:, arr_atlas != 0]
    equins_edge = equins[:, arr_atlas != 0]

    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True).fit(
        temoins_edge.reshape(temoins_edge.shape[0], temoins_edge.shape[1] * 3))

    decomp_t = kpca.transform(temoins_edge.reshape(temoins_edge.shape[0], temoins_edge.shape[1] * 3))
    recons_t = kpca.inverse_transform(decomp_t).reshape(temoins_edge.shape[0], temoins_edge.shape[1], 3)
    p_t = np.power((recons_t - temoins_edge), 2)
    p_t = np.mean(p_t, axis=2)
    reconstruction_t = np.zeros((temoins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2], 3))
    reconstruction_t[:, arr_atlas != 0] = recons_t
    detection_t = np.zeros((temoins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2]))
    detection_t[:, arr_atlas != 0] = p_t

    decomp_e = kpca.transform(equins_edge.reshape(equins_edge.shape[0], equins_edge.shape[1] * 3))
    recons_e = kpca.inverse_transform(decomp_e).reshape(equins_edge.shape[0], equins_edge.shape[1], 3)
    p = np.power((recons_e - equins_edge), 2)
    p = np.mean(p, axis=2)
    reconstruction_e = np.zeros((equins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2], 3))
    reconstruction_e[:, arr_atlas != 0] = recons_e
    detection_e = np.zeros((equins.shape[0], arr_atlas.shape[0], arr_atlas.shape[1], arr_atlas.shape[2]))
    detection_e[:, arr_atlas != 0] = p

    print('KPCA finished')
    return detection_e, detection_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Texture plugger')
    parser.add_argument('-b', '--bone', help='bone to analysis', type=str, required=True, default='calcaneus')
    parser.add_argument('-t', '--td', help='path to TD subjects\' Warp files', type=str, required=True, default='./td_warp')
    parser.add_argument('-c', '--cp', help='path to CP subjects\' Warp files', type=str, required=True, default='./cp_warp')
    parser.add_argument('-o', '--output', help='path to output files', type=str, required=False, default='./results')

    args = parser.parse_args()

    bone = args.bone
    method = ['t2', 'kpca', 'pca']
    def_td = args.td
    def_cp = args.cp

    atlas = './' + bone + '.nii.gz'
    img_atlas = nib.load(atlas)
    affine = img_atlas.affine
    arr_atlas = img_atlas.get_fdata()
    arr_atlas = dilation(arr_atlas)
    shape = arr_atlas.shape

    warp_shape = (shape[0], shape[1], shape[2], 3)

    deformation_temoin = np.zeros(warp_shape, dtype=float)
    deformation_temoin = deformation_temoin[np.newaxis, :]

    deformation_equin = np.zeros(warp_shape, dtype=float)
    deformation_equin = deformation_equin[np.newaxis, :]

    temoins = glob.glob(os.path.join(def_td, '*1Warp.nii.gz'))
    equins = glob.glob(os.path.join(def_cp, '*1Warp.nii.gz'))

    thr = Pool(processes=5)
    for t in range(len(temoins)):
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

    result_path = args.output
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    min_max_scaler = preprocessing.MinMaxScaler()

    score = np.zeros(shape)

    for m in method:
        if m == 'kpca':
            p, p_t = detection_kpca(deformation_temoin, deformation_equin, arr_atlas)
        elif m == 't2':
            p, p_t = detection_t2(deformation_temoin, deformation_equin, arr_atlas)
        else:
            p, p_t = detection_pca(deformation_temoin, deformation_equin, arr_atlas)

        z = np.zeros((len(equins), shape[0], shape[1], shape[2]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if arr_atlas[i, j, k] == 0:
                        pass
                    else:
                        z[:, i, j, k] = stats.zmap(scores=p[:, i, j, k], compare=p_t[:, i, j, k])
        z = np.abs(z)

        score_m = np.zeros(shape)

        score_m = min_max_scaler.fit_transform(score_m.reshape(-1, 1)).reshape(shape)
        score = score + score_m
        img = nib.Nifti1Image(score_m, affine)
        nib.save(img, os.path.join(result_path, bone + '_score_' + m + '.nii.gz'))

    score = score / 3
    score = min_max_scaler.fit_transform(score.reshape(-1, 1)).reshape(shape)
    img = nib.Nifti1Image(score, affine)
    nib.save(img,
             os.path.join(result_path, bone + '_anomap.nii.gz'))





