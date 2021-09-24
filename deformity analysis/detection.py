import os
import numpy as np
import nibabel as nib
from scipy import stats
from multiprocessing import Pool
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from pyanom.outlier_detection import HotelingT2


def data_loading(input_path):
    img = nib.load(input_path)
    img_array = img.get_fdata().astype(float)
    return img_array.flatten()


def img_generation(array, affine, save_path):
    img = nib.Nifti1Image(array, affine)
    nib.save(img, save_path)


def detection_t2(temoins, equins, nums, nums_t, arr_atlas, affine, detection_save_path):
    p = np.zeros((equins.shape[0], 323, 323, 144, 3), dtype=float)
    p_t = np.zeros((temoins.shape[0], 323, 323, 144, 3), dtype=float)
    print('Starting Hotelling\'s T2 test')

    for i in range(107):
        for j in range(107):
            for k in range(48):
                if (arr_atlas[i*3:(i+1)*3, j*3:(j+1)*3, k*3:(k+1)*3] == np.zeros((3, 3, 3, 3))).all():
                    pass
                else:
                    for m in range(3):
                        model = HotelingT2()
                        model.fit((temoins[:, i*3:(i+1)*3, j*3:(j+1)*3, k*3:(k+1)*3, m].reshape(-1, 1)))
                        temoin_score = model.score(temoins[:, i*3:(i+1)*3, j*3:(j+1)*3, k*3:(k+1)*3, m].reshape(-1, 1))
                        p_t[:, i * 3:(i + 1) * 3, j * 3:(j + 1) * 3, k * 3:(k + 1) * 3, m] = temoin_score.reshape(temoins.shape[0], 3, 3, 3)
                        anomaly_score = model.score(equins[:, i*3:(i+1)*3, j*3:(j+1)*3, k*3:(k+1)*3, m].reshape(-1, 1))
                        p[:, i * 3:(i + 1) * 3, j * 3:(j + 1) * 3, k * 3:(k + 1) * 3, m] = anomaly_score.reshape(equins.shape[0], 3, 3, 3)
    p = np.mean(p, axis=4)
    p_t = np.mean(p_t, axis=4)

    for i in range(len(nums)):
        img_generation(p_t[i], affine, os.path.join(detection_save_path, nums_t[i] + '_score_0.nii.gz'))
        img_generation(p[i], affine, os.path.join(detection_save_path, nums[i] + '_score_0.nii.gz'))

    print('Hotelling\'s T2 test finished')
    return p, p_t


def detection_kpca(temoins, equins, nums, nums_t, affine, detection_save_path):
    print('Starting KPCA')

    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True).fit(temoins.reshape(temoins.shape[0], 323 * 323 * 144 * 3))
    
    decomp_t = kpca.transform(temoins.reshape(temoins.shape[0], 323 * 323 * 144 * 3))
    recons_t = kpca.inverse_transform(decomp_t).reshape(temoins.shape[0], 323,  323, 144, 3)
    p_t = np.power((recons_t - temoins), 2)
    p_t = np.mean(p_t, axis=4)

    for i in range(len(nums_t)):
        img_generation(recons_t[i], affine, os.path.join(detection_save_path, nums_t[i] + '_reconstruction.nii.gz'))
        img_generation(p_t[i], affine, os.path.join(detection_save_path, nums_t[i] + '_score_0.nii.gz'))
    
    decomp_e = kpca.transform(equins.reshape(equins.shape[0], 323 * 323 * 144 * 3))
    recons_e = kpca.inverse_transform(decomp_e).reshape(equins.shape[0], 323,  323, 144, 3)
    p = np.power((recons_e - equins), 2)
    p = np.mean(p, axis=4)

    for i in range(len(nums)):
        img_generation(recons_e[i], affine, os.path.join(detection_save_path, nums[i] + '_reconstruction.nii.gz'))
        img_generation(p[i], affine, os.path.join(detection_save_path, nums[i] + '_score_0.nii.gz'))

    print('KPCA finished')
    return p, p_t


if __name__ == '__main__':
    bone = 'calcaneus'
    method = ['kpca', 't2']
    deformationDirectory = './' + bone + '/deformation_edge'
    out_path = './' + bone
    atlas = './atlas_unet3_edge_' + bone + '.nii.gz'

    img_atlas = nib.load(atlas)
    affine = img_atlas.affine
    arr_atlas = img_atlas.get_fdata()

    temoin_num = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T11']
    equin_num = ['E01', 'E02', 'E05', 'E06', 'E08', 'E09', 'E10', 'E11', 'E13']
    #equin_num = ['equins']
    shape = (323, 323, 144, 3)

    deformation_temoin = np.zeros(shape, dtype=float)
    deformation_temoin = deformation_temoin[np.newaxis, :]

    deformation_equin = np.zeros(shape, dtype=float)
    deformation_equin = deformation_equin[np.newaxis, :]

    thr = Pool(processes=5)
    for t in range(len(temoin_num)):
        deformation_t = thr.apply_async(data_loading, args=(os.path.join(deformationDirectory, temoin_num[t] + '_edge.nii.gz'),))
        if t == 0:
            deformation_temoin[0] = deformation_t.get().reshape(shape)
        else:
            deformation_temoin = np.insert(deformation_temoin, t, values=deformation_t.get().reshape(shape), axis=0)

    for e in range(len(equin_num)):
        deformation_e = thr.apply_async(data_loading, args=(os.path.join(deformationDirectory, equin_num[e] + '_edge.nii.gz'),))
        if e == 0:
            deformation_equin[0] = deformation_e.get().reshape(shape)
        else:
            deformation_equin = np.insert(deformation_equin, e, values=deformation_e.get().reshape(shape), axis=0)

    thr.close()
    thr.join()
    print('data loaded')

    score = np.zeros((323, 323, 144))
    result_path = os.path.join(out_path, 'results')
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    for m in method:
        detection_save_path = os.path.join(out_path, m)
        if not os.path.isdir(detection_save_path):
            os.mkdir(detection_save_path)
        if m == 'kpca':
            p, p_t = detection_kpca(deformation_temoin, deformation_equin, equin_num, temoin_num, affine, detection_save_path)
        elif m == 't2':
            p, p_t = detection_t2(deformation_temoin, deformation_equin, equin_num, temoin_num, arr_atlas, affine, detection_save_path)

        z = np.zeros((len(equin_num), 323, 323, 144))
        for i in range(323):
            for j in range(323):
                for k in range(144):
                    if arr_atlas[i, j, k] == 0:
                        pass
                    else:
                        z[:, i, j, k] = stats.zmap(scores=p[:, i, j, k], compare=p_t[:, i, j, k])
        z = np.abs(z)

        score_m = np.zeros((323, 323, 144))
        for i in range(len(equin_num)):
            img_generation(z[i], affine, os.path.join(detection_save_path, equin_num[i] + '_score_1.nii.gz'))
            score_m = score_m + z[i]
        min_max_scaler = preprocessing.MinMaxScaler()
        score_m = min_max_scaler.fit_transform(score_m.reshape(-1, 1)).reshape(323, 323, 144)
        score = score + score_m

        img = nib.Nifti1Image(score_m, affine)
        nib.save(img, os.path.join(result_path, bone + '_score_' + m + '.nii.gz'))

    img = nib.Nifti1Image(score, affine)
    nib.save(img,
             os.path.join(result_path, bone + '_anomap.nii.gz'))

