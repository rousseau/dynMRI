#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:59:19 2021

@author: benjamin
"""

import itk
import numpy as np
import meshio
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Texture plugger')
  parser.add_argument('-is', '--inputSTL', help='Input STL', type=str, required=False, default='./calcaneus.stl')
  parser.add_argument('-in', '--inputNifti', help='Input nifti texture', type=str, required=False, default='./results/calcaneus_anomap.nii')
  parser.add_argument('-o', '--output', help='Output mesh', type=str, required=False, default='./results/calcaneus_anomap.vtk')
  parser.add_argument('-s', '--smoothing', help ='Smoothing type, nearest, linear or BSpline', type=str, required=False, default='BSpline')
  parser.add_argument('-tn', '--texture_name', help='Custom name for added texture', type=str, required=False, default='deformation')

  args = parser.parse_args()

stl_path = args.inputSTL
nii_path = args.inputNifti
smoothing_type = args.smoothing
output_path = args.output
texture_name = args.texture_name

mesh = meshio.read(stl_path)
itk_image = itk.imread (nii_path)

if smoothing_type == 'linear':
    texture = np.zeros (len(mesh.points))
    lin_interpolator = itk.LinearInterpolateImageFunction.New(itk_image)
    for i in range(len(mesh.points)):
        #find the closest pixel
        index = itk_image.TransformPhysicalPointToContinuousIndex((float(mesh.points[i][0]), float(mesh.points[i][1]), float(mesh.points[i][2])))
        #get the texture from the pixel
        texture[i] = lin_interpolator.EvaluateAtContinuousIndex(index)

if smoothing_type == 'BSpline':
    #BSpline interpolator
    texture = np.zeros (len(mesh.points))
    BSpline_interpolator = itk.BSplineInterpolateImageFunction.New(itk_image)
    for i in range(len(mesh.points)):
        #find the closest pixel
        index = itk_image.TransformPhysicalPointToContinuousIndex((float(mesh.points[i][0]), float(mesh.points[i][1]), float(mesh.points[i][2])))
        #get the texture from the pixel
        texture[i] = BSpline_interpolator.EvaluateAtContinuousIndex(index)

if smoothing_type == 'NearestNeighbor':
    # for binarized images
    texture = np.zeros (len(mesh.points))
    nn_interpolator = itk.NearestNeighborInterpolateImageFunction.New(itk_image)
    for i in range(len(mesh.points)):
        #find the closest pixel
        index = itk_image.TransformPhysicalPointToContinuousIndex((float(mesh.points[i][0]), float(mesh.points[i][1]), float(mesh.points[i][2])))
        #get the texture from the pixel
        texture[i] = nn_interpolator.EvaluateAtContinuousIndex(index)
        
mesh.point_data = {texture_name: texture}
mesh.write(output_path)
