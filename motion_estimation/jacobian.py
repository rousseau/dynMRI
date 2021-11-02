# -*- coding: utf-8 -*-

"""
  © IMT Atlantique - LATIM-INSERM UMR 1101
  Author(s): Karim Makki (karim.makki@imt-atlantique.fr)

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.
"""

import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage.filters import gaussian_filter


def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())

def nifti_get_affine(filename):

    nii = nib.load(filename)

    return (nii.affine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dfield', '--deffield', help='4 dimensional volume, each 3D image in the volume contains the \
    coordinates of the displacement field with respect to one direction in the space, by convention the first image \
    contains coordinates of the x_axis, the second image contains coordinates of the y_axis and the third one contains\
    coordinates of the z_axis', type=str, required  = True)
    parser.add_argument('-o', '--output', help='Output image full path', type=str, default='jacobian_map.nii.gz')
    parser.add_argument('-wfield', '--warpfield', help='0: if you treat warp field as relative: y = x + w(x), so \
    dy/dx = 1 + dw(x)/dx. 1: if you treat warp field as absolute: y = w(x) so dy/dx = dw(x)/dx ', type=int, default=1)

    args = parser.parse_args()

    ###Load the displacement field w(x) = x'-x where x is the initial position and x' is the target position of the voxel

    def_field = nifti_to_array(args.deffield)

    #### Compute jacobian matrix of the displacement field J = dw(x)/dx

    gx_x,gx_y,gx_z = np.gradient(def_field[...,0])
    gy_x,gy_y,gy_z = np.gradient(def_field[...,1])
    gz_x,gz_y,gz_z = np.gradient(def_field[...,2])
   
    
    if args.warpfield == 0 :
       gx_x +=  1
       gy_y +=  1
       gz_z +=  1

    ### Gradient vector field Smoothing 

    gaussian_filter(gx_x, sigma=2, output=gx_x)
    gaussian_filter(gx_y, sigma=2, output=gx_y)
    gaussian_filter(gx_z, sigma=2, output=gx_z)
    
    gaussian_filter(gy_x, sigma=2, output=gy_x)
    gaussian_filter(gy_y, sigma=2, output=gy_y)
    gaussian_filter(gy_z, sigma=2, output=gy_z)
    
    gaussian_filter(gz_x, sigma=2, output=gz_x)
    gaussian_filter(gz_y, sigma=2, output=gz_y)
    gaussian_filter(gz_z, sigma=2, output=gz_z)

    #### Compute determinant using the rule of Sarrus

    Jacobian = ( (gx_x*gy_y*gz_z) + (gy_x*gz_y*gx_z) + (gz_x*gx_y*gy_z) ) - ( (gz_x*gy_y*gx_z) + (gy_x*gx_y*gz_z) + (gx_x*gz_y*gy_z) )


    #### Save the jacobian as a 3D nifti file

    j = nib.Nifti1Image(Jacobian, nifti_get_affine(args.deffield))
    save_path = args.output
    nib.save(j, save_path)
    del def_field
