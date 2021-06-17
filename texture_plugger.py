#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:08:41 2021

This script opens a STL and a corresponding nifti and plug the texture of the nifti on the STL, then saves it to a ply format viewable in paraview.
Texture values are converted on a 0-255 scale respecting the original spread and ratio of values

TODO:
-find a PLY->gifti converter

Try using nibabel instead of sitk
add the texture as a vtk

Input: STL and nifti
Output: PLY first

@author: benjamin
"""

import matplotlib.pyplot as plt
import vtk
import itk
import SimpleITK as sitk
import numpy as np
import nibabel as nb
import trimesh as tr
from scipy import stats as sps

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Texture plugger')
  parser.add_argument('-is', '--inputSTL', help='Input STL', type=str, required=False, default = '/home/benjamin/Documents/DataYue/atlas_calcaneus_segment_full.stl')
  parser.add_argument('-it', '--inputNifti', help='Input nifti texture', type=str, required=False, default = '/home/benjamin/Documents/DataYue/score_deformation_edge_all.nii')
  parser.add_argument('-o', '--output', help='Output ply mesh', type=str, required=False, default = 'lin_mesh.ply')
  parser.add_argument('-s', '--smoothing', help = 'Smoothing type, linear or Bspline', type = str, required = False, default = 'BSpline')

  args = parser.parse_args()

stl_path = args.inputSTL
nii_path = args.inputNifti
smoothing_type = args.smoothing

mesh = tr.load(stl_path)
itkimage = itk.imread (nii_path)

if smoothing_type == 'linear':
    texture = np.zeros (len(mesh.vertices))
    lin_interpolator = itk.LinearInterpolateImageFunction.New (itkimage)
    for i in range(len(mesh.vertices)):
        #find the closest pixel
        index = itkimage.TransformPhysicalPointToContinuousIndex(mesh.vertices[i])
        #get the texture from the pixel
        texture[i] = lin_interpolator.EvaluateAtContinuousIndex(index)

if smoothing_type == 'BSpline':
    #BSpline interpolator
    texture = np.zeros (len(mesh.vertices))
    BSpline_interpolator = itk.BSplineInterpolateImageFunction.New (itkimage)
    for i in range(len(mesh.vertices)):
        #find the closest pixel
        index = itkimage.TransformPhysicalPointToContinuousIndex(mesh.vertices[i])
        #get the texture from the pixel
        texture[i] = BSpline_interpolator.EvaluateAtContinuousIndex(index)

def uglyRGB (texture):
    #include converstion to 0-255 scale
    output = np.zeros (len(texture))
    maxtex = max(texture)
    mintex = min(texture)
    for i in range(len(texture)):
        if i == 0:
            output[i] = mintex
        else:
            output[i] = ((texture[i] - mintex) * 255) / (maxtex-mintex) + mintex
    return output
 
    
texture = uglyRGB(texture)   
#modify in place texture of a mesh

for i in range(len(mesh.visual.vertex_colors)):
    mesh.visual.vertex_colors[i] = [texture[i], texture[i], texture[i], texture[i]] #red, green, blue, alpha (?)
    
mesh.show()

mesh.export(args.output)
   
