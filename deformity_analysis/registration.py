#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: CYY
"""

import os
import glob
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_cdt


def generation_fuzzy_dist(file):
    img = nib.load(file)
    imdata = img.get_fdata().astype(float)
    segbin = np.clip(imdata, 0, 1)
    segfuz = gaussian_filter(segbin, 1)
    nib.save(nib.Nifti1Image(segfuz, img.affine), file.split('.nii')[0] + '_fuzzy.nii.gz')

    segdist = gaussian_filter(distance_transform_cdt(segbin), 1)
    segdist = segdist * 1.0 / np.max(segdist)
    nib.save(nib.Nifti1Image(segdist, img.affine), file.split('.nii')[0] + '_dist.nii.gz')


def registration(fixedMask, fixedDist, movingMask, movingDist, outputPrefix_image):
    registration = 'antsRegistration -d 3 ' + \
                   '-m MeanSquares[' + fixedMask + ',' + movingMask + ',1,0] ' + \
                   '-m MeanSquares[' + fixedDist + ',' + movingDist + ',1,0] ' + \
                   '-t Rigid[0.1] -f 6x4x2x1 -s 3x2x1x0 -c 100x100x70x20 ' + \
                   '-m MeanSquares[' + fixedMask + ',' + movingMask + ',1,0] ' + \
                   '-m MeanSquares[' + fixedDist + ',' + movingDist + ',1,0] ' + \
                   '-t Affine[0.1] -f 6x4x2x1 -s 3x2x1x0 -c 100x100x70x20 ' + \
                   '-m MeanSquares[' + fixedMask + ',' + movingMask + ',1,0] ' + \
                   '-m MeanSquares[' + fixedDist + ',' + movingDist + ',1,0] ' + \
                   '-t SyN[0.1] -f 6x4x2x1 -s 3x2x1x0 -c 100x100x70x20 -v 1 ' + \
                   '-o [' + outputPrefix_image + ',' + outputPrefix_image + 'Warped.nii.gz,' + outputPrefix_image + 'InverseWarped.nii.gz]'

    print(registration)
    os.system(registration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Texture plugger')
    parser.add_argument('-n', '--thread', help='number of thread for calculation', type=str, required=False, default='12')
    parser.add_argument('-t', '--td', help='path to TD segmention files', type=str, required=False, default='./data/td')
    parser.add_argument('-c', '--cp', help='path to CP segmention files', type=str, required=False, default='./data/cp')
    parser.add_argument('-ot', '--output_td', help='path to output files of TD', type=str, required=False, default='./td_warp')
    parser.add_argument('-oc', '--output_cp', help='path to output files of CP', type=str, required=False, default='./cp_warp')

    args = parser.parse_args()
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = args.thread
    td_path = args.td
    cp_path = args.cp
    output_td = args.output_td
    output_cp = args.output_cp

    if not os.path.isdir(output_cp):
        os.makedirs(output_cp)
    if not os.path.isdir(output_td):
        os.makedirs(output_td)

    template = './template_label.nii.gz'
    generation_fuzzy_dist(template)
    fixedMask = './template_fuzzy.nii.gz'
    fixedDist = './template_dist.nii.gz'

    temoins = glob.glob(os.path.join(td_path, '*_seg.nii.gz'))
    equins = glob.glob(os.path.join(cp_path, '*_seg.nii.gz'))
    for t in temoins:
        generation_fuzzy_dist(t)
        movingMask = t.split('.nii')[0] + '_fuzzy.nii.gz'
        movingDist = t.split('.nii')[0] + '_dist.nii.gz'
        outputPrefix = os.path.join(output_td, t.rsplit('/', 1)[1].split('seg')[0])
        registration(fixedMask, fixedDist, movingMask, movingDist, outputPrefix)
    for e in equins:
        generation_fuzzy_dist(e)
        movingMask = e.split('.nii')[0] + '_fuzzy.nii.gz'
        movingDist = e.split('.nii')[0] + '_dist.nii.gz'
        outputPrefix = os.path.join(output_cp, e.rsplit('/', 1)[1].split('seg')[0])
        registration(fixedMask, fixedDist, movingMask, movingDist, outputPrefix)





