#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:08:41 2021

This script opens a STL and a corresponding nifti and plug the texture of the nifti on the STL, and saves it to a ply format viewable in paraview.
Texture values are converted on a 0-255 scale respecting the original spread and ratio of values

TODO:
-find a PLY->gifti converter

Try using nibabel instead of sitk
add the texture as a vtk

Input: STL and nifti
Output: PLY first

@author: benjamin, yue
"""

import itk
import numpy as np
import trimesh as tr
from sklearn import preprocessing
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Texture plugger')
  parser.add_argument('-is', '--inputSTL', help='Input STL', type=str, required=False, default = '/home/benjamin/Documents/DataYue/atlas_calcaneus_segment_full.stl')
  parser.add_argument('-it', '--inputNifti', help='Input nifti texture', type=str, required=False, default = '/home/benjamin/Documents/DataYue/score_deformation_edge_all.nii')
  parser.add_argument('-o', '--output', help='Output ply mesh', type=str, required=False, default = 'lin_mesh.ply')
  parser.add_argument('-s', '--smoothing', help = 'Smoothing type, NearestNeighbor, linear or Bspline', type = str, required = False, default = 'BSpline')

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

if smoothing_type == 'NearestNeighbor':
    # for binarized images
    texture = np.zeros (len(mesh.vertices))
    nn_interpolator = itk.NearestNeighborInterpolateImageFunction.New (itkimage)
    for i in range(len(mesh.vertices)):
        #find the closest pixel
        index = itkimage.TransformPhysicalPointToContinuousIndex(mesh.vertices[i])
        #get the texture from the pixel
        texture[i] = nn_interpolator.EvaluateAtContinuousIndex(index)


def uglyRGB (texture):
    #include converstion to 0-255 scale
    #output = np.zeros (len(texture))
    min_max_scaler = preprocessing.MinMaxScaler()
    output = min_max_scaler.fit_transform(texture.reshape(-1, 1)).reshape(len(texture)) * 255
    '''
    maxtex = max(texture)
    print(maxtex)
    mintex = min(texture)
    print(mintex)
    for i in range(len(texture)):
        if i == 0:
            output[i] = mintex
        else:
            output[i] = ((texture[i] - mintex) * 255) / (maxtex-mintex) + mintex
    '''
    return output
 

#texture = uglyRGB(texture)
#modify in place texture of a mesh
mesh.visual.vertex_colors
for i in range(len(mesh.visual.vertex_colors)):
    mesh.visual.vertex_colors[i] = [texture[i], texture[i], texture[i], texture[i]] #red, green, blue, alpha (?)

#mesh.show()

mesh.export(args.output)
    
########
#LEGACY#
########
'''
Grayscale->RGB elegant conversion: int rgb = grey * 0x00010101;

convert to new scale
OldRange = (OldMax - OldMin)
if (OldRange == 0)
    NewValue = NewMin
else
{
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
}

# #no interpolation
# texture = np.zeros (len(mesh.vertices))
# for i in range(len(mesh.vertices)):
#     #find the closest pixel
#     index = sitkimage.TransformPhysicalPointToIndex (mesh.vertices[i])
#     #get the texture from the pixel
#     texture [i] = sitkimage.GetPixel (index)


#load a vtk using vtk (not possible using meshio as unstructured)
reader = vtk.vtkStructuredPointsReader()
reader.SetFileName(heatmap)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()

data = reader.GetOutput()

#from sitk image, get bounding boxes (not verified)
stats = sitk.StatisticsImageFilter ()
stats.Execute (itkimage)
stats.GetMaximum ()

mask = itkimage > 1

stats = sitk.LabelStatisticsImageFilter ()
stats.Execute (itkimage, mask)

stats.GetBoundingBox(1) #is the label correct ? How to verify


###nifti inline viewer
def myshow(img, title=None, margin=0.05):
    
    if (img.GetDimension() == 3):
        img = sitk.Tile( (img[img.GetSize()[0]//2,:,:],
                          img[:,img.GetSize()[1]//2,:],
                          img[:,:,img.GetSize()[2]//2]), [2,2])
            
    
    aimg = sitk.GetArrayViewFromImage(img)
    
    xsize,ysize = aimg.shape

    dpi=80
    
    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1+margin)*ysize / dpi, (1+margin)*xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    t = ax.imshow(aimg)
    if len(aimg.shape) == 2:
        t.set_cmap("gray")
    if(title):
        plt.title(title)
    plt.show()

'''
