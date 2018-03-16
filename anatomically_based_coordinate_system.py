# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA

import SimpleITK as sitk
import nibabel as nib
from numpy.linalg import inv
from numpy.linalg import det



def Text_file_to_matrix(filename):

   T = np.loadtxt(str(filename), dtype='f')

   return np.mat(T)


image='/home/karim/Data/subject05/20161020_1227183DT1EGSENSEs2601a1026.nii.gz'
dynamic='/home/karim/Data/sitk/dyn0000.nii.gz'

ref_matrix_calcaneus= '/home/karim/Exp/November/Mardi_28_/healthy_subjects_new/subject05/20161020_122718MOVIECLEARSAGPASSIVESENSEs1201a1012/propagation/output_path_component0/final_results/direct_static_on_dyn0007_component_0.mat'
ref_matrix_talus= '/home/karim/Exp/November/Mardi_28_/healthy_subjects_new/subject05/20161020_122718MOVIECLEARSAGPASSIVESENSEs1201a1012/propagation/output_path_component1/final_results/direct_static_on_dyn0007_component_1.mat'
ref_matrix_tibia= '/home/karim/Exp/November/Mardi_28_/healthy_subjects_new/subject05/20161020_122718MOVIECLEARSAGPASSIVESENSEs1201a1012/propagation/output_path_component2/final_results/direct_static_on_dyn0007_component_2.mat'


a= im1.GetOrigin()
b= dyn.GetOrigin()

nii = nib.load(image)



def warp_point_using_flirt_transform(point,input_header, reference_header, transform):
# flip [x,y,z] if necessary (based on the sign of the sform or qform determinant)
	if np.sign(det(input_header.get_sform()))==1:
		point[0] = input_header.get_data_shape()[0]-1-point[0]
		point[1] = input_header.get_data_shape()[1]-1-point[1]
		point[2] = input_header.get_data_shape()[2]-1-point[2]
#scale the values by multiplying by the corresponding voxel sizes (in mm)
	p=np.ones((4))
    #point = np.ones(4)
	p[0] = point[0]*input_header.get_zooms()[0]
	p[1] = point[1]*input_header.get_zooms()[1]
	p[2] = point[2]*input_header.get_zooms()[2]
# apply the FLIRT matrix to map to the reference space
	p = np.dot(transform, p[:,np.newaxis])
    #print(point.shape)

#divide by the corresponding voxel sizes (in mm, of the reference image this time)
	p[0, np.newaxis] /= reference_header.get_zooms()[0]
	p[1, np.newaxis] /= reference_header.get_zooms()[1]
	p[2, np.newaxis] /= reference_header.get_zooms()[2]

#flip the [x,y,z] coordinates (based on the sign of the sform or qform determinant, of the reference image this time)
	if np.sign(det(reference_header.get_sform()))==1:
		p[0, np.newaxis] = reference_header.get_data_shape()[0]-1-p[0, np.newaxis]
		p[1, np.newaxis] = reference_header.get_data_shape()[1]-1-p[1, np.newaxis]
		p[2, np.newaxis] = reference_header.get_data_shape()[2]-1-p[2, np.newaxis]

	return np.absolute(np.delete(p, 3, 0))


'''
Given the  XYZ orthogonal coordinate system (image coordinate system), find a transformation, M, that maps XYZ  to an anatomical orthogonal coordinate system UVW
.
'''
## Compute the change of basis matrix to go from the image coordinate system to one locally defined bone coordinate system

def Image_to_bone_coordinate_system(image,U,V,W,bone_origin=None):

    nii = nib.load(image)

    M = np.identity(4)
    # Rotation bloc
    M[0][0] = U[0]
    M[1][0] = U[1]
    M[2][0] = U[2]
    M[0][1] = V[0]
    M[1][1] = V[1]
    M[2][1] = V[2]
    M[0][2] = W[0]
    M[1][2] = W[1]
    M[2][2] = W[2]

    # Translation bloc: here, origin coordinates must be expressed in mm

    image_origin = sitk.ReadImage(image).GetOrigin()

    if bone_origin is not None:

        #express origin coordinates in mm

        M[0][3] = bone_origin[0]*nii.header.get_zooms()[0] - image_origin[0]
        M[1][3] = bone_origin[1]*nii.header.get_zooms()[1] - image_origin[1]
        M[2][3] = bone_origin[2]*nii.header.get_zooms()[2] - image_origin[2]
   # else: return only rotations

    return(M)



'''
>>> Ti: 4*4 flirt transformation matrix expressed in the image coordinate system
>>> M: 4*4 change of basis transformation matrix (from )

[u',v',w',1] = M*T*M^-1 * [u,v,w,1]
'''

def Express_transformation_matrix_in_bone_coordinate_system(Ti,Mi):

    return(np.dot(np.dot(Mi,Ti),inv(Mi)))





'''
>>> Anatomically based coordinate system:
>>> calcaneal coordinate system
>>> the calcaneal x-axis(cx) was defined as the unit vector connecting the most anterior-inferior and the posterior-inferior
     calcaneal points, i.e. define the calcaneal x-axis (cx) from point0 and point1
>>> In general the x-axis was directed to the left, the y-axis was directed superiorly and the z-axis was directly anteriorly.
>>> the temporary calcaneal z-axis(∼cz) was defined as the unit vector connecting the insertion of the long plantar ligament
     (also known as the calcaneal origin,(Co) and the most convex point of the posterior-lateral curve of the calcaneus), i.e,
     define the calcaneal z-axis(cz) from point3 and point4
>>> cross products were used to create orthogonal coordinate system. i.e, to determine the calcaneal y-axis (cy)
>>> point1 : the most posterior-inferior calcaneal point
>>> point2 : the most anterior-inferior calcaneal point
>>> point3 : the calcaneal origin "Co", (see sheehan's paper for visual details)
>>>
'''


'''####### Define the calcaneal coordinate system ##################'''

#the calcaneal x-axis(cx) was defined as the unit vector connecting the most anterior-inferior and the posterior-inferior calcaneal point.

## point0 : the most posterior-inferior calcaneal point
point0 = np.zeros([3,1])
point0[0] = 456 #int(raw_input("Please enter the x_coordinate of the most posterior-inferior calcaneal point: "))-1
point0[1] = 187 #int(raw_input("Please enter the y_coordinate of the most posterior-inferior calcaneal point: "))-1
point0[2] = 93 #int(raw_input("Please enter the z_coordinate of the most posterior-inferior calcaneal point: "))-1
## map the point into the neutral position
point0= warp_point_using_flirt_transform(point0,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))

## point1 : the most anterior-inferior calcaneal point
point1 = np.zeros([3,1])
point1[0] = 291#int(raw_input("Please enter the x_coordinate of the most anterior-inferior calcaneal point: "))-1
point1[1] = 136#int(raw_input("Please enter the y_coordinate of the most anterior-inferior calcaneal point: "))-1
point1[2] = 93#int(raw_input("Please enter the z_coordinate of the most anterior-inferior calcaneal point: "))-1
point1= warp_point_using_flirt_transform(point1,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))

cx = (point1-point0)/LA.norm(point1-point0)

## point2 : the most convex point of the posterior-lateral curve of the calcaneus
point2= np.zeros([3,1])
point2[0] = 463#int(raw_input("Please enter the x_coordinate of the most convex point of the posterior-lateral curve of the calcaneus: "))-1
point2[1] = 241#int(raw_input("Please enter the y_coordinate of the most convex point of the posterior-lateral curve of the calcaneus: "))-1
point2[2] = 81#int(raw_input("Please enter the z_coordinate of the the most convex point of the posterior-lateral curve of the calcaneus: "))-1
point2= warp_point_using_flirt_transform(point2,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))

## point3 : the insertion of the long plantar ligament (also known as the calcaneal origin,(Co))
point3= np.zeros([3,1])
point3[0] = 430#int(raw_input("Please enter the x_coordinate of the insertion of the long plantar ligament: "))-1
point3[1] = 241#int(raw_input("Please enter the y_coordinate of the insertion of the long plantar ligament: "))-1
point3[2] = 131#int(raw_input("Please enter the z_coordinate of the insertion of the long plantar ligament : "))-1
point3= warp_point_using_flirt_transform(point3,nii.header, nii.header, Text_file_to_matrix(ref_matrix_calcaneus))


###The temporary calcaneal z-axis(∼cz_t)was defined as the unit vector connecting the insertion of the long plantar ligament (also known as
###the calcaneal origin,(Co) and the most convex point of the posterior-lateral curve of the calcaneus.

cz_t = (point3-point2)/LA.norm(point3-point2)

# The cross product of cx and cz in R^3 is a vector perpendicular to both cx and cz
###For the calcaneus, cy was defined as(∼cz×cx)

cy = np.cross(cz_t, cx, axis=0)
cy/= LA.norm(cy)

###For the calcaneus, cz was defined as(cx×cy)

cz = np.cross(cx,cy, axis=0)
cz/= LA.norm(cz)

'''####### Define the talar coordinate system ##################
>>> Anatomically based coordinate system:
>>> the talar x-axis(ax) was defined as the unit vector that bisected the arc formed by the two lines connecting the talar
sinus with the most anterior-superior and anterior-inferior talar points.
'''

#talar sinus point
point4 = np.zeros([3,1])
point4[0] = 229#int(raw_input("Please enter the x_coordinate of the talar sinus point: "))-1
point4[1] = 256#int(raw_input("Please enter the y_coordinate of the talar sinus point: "))-1
point4[2] = 100#int(raw_input("Please enter the z_coordinate of talar sinus point: "))-1
point4= warp_point_using_flirt_transform(point4,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

#most anterior-superior talar point
point5 = np.zeros([3,1])
point5[0] = 137#int(raw_input("Please enter the x_coordinate of the most anterior-superior talar point: "))-1
point5[1] = 229#int(raw_input("Please enter the y_coordinate of the most anterior-superior talar point: "))-1
point5[2] = 100#int(raw_input("Please enter the z_coordinate of of the most anterior-superior talar point: "))-1
point5= warp_point_using_flirt_transform(point5,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

#most anterior-inferior talar point
point6 = np.zeros([3,1])
point6[0] = 191#int(raw_input("Please enter the x_coordinate of the most anterior-inferior talar point: "))-1
point6[1] = 188#int(raw_input("Please enter the y_coordinate of the most anterior-inferior talar point: "))-1
point6[2] = 100#int(raw_input("Please enter the z_coordinate of of the most anterior-inferior talar point: "))-1
point6= warp_point_using_flirt_transform(point6,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))


ax= (point5-point4)*LA.norm(point6-point4) + (point6-point4)*LA.norm(point5-point4)
ax/= LA.norm(ax)
#####################################################################################

##>>> the temporary talar y-axis(∼ay) was defined as the line that bisected the arc formed by the triangle defining the distal
##talar surface directly inferior to the talar dome. Ao was the most inferior point on the talar dome section of the talus.
#Ao: the most inferior point on the talar dome section of the talus

point7 = np.zeros([3,1])
point7[0] = 251#int(raw_input("Please enter the x_coordinate of the most inferior point on the talar dome section of the talus: "))-1
point7[1] = 256#int(raw_input("Please enter the y_coordinate of the most inferior point on the talar dome section of the talus: "))-1
point7[2] = 100#int(raw_input("Please enter the z_coordinate of the most inferior point on the talar dome section of the talus: "))-1
point7= warp_point_using_flirt_transform(point7,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

#Defining the distal talar surface directly inferior to the talar dome:

#point A:
point8 = np.zeros([3,1])
point8[0] = 306 #int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the x_coordinate of the point A: "))-1
point8[1] = 311#int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the y_coordinate of the point A: "))-1
point8[2] = 100#int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the z_coordinate of the point A: "))-1
point8= warp_point_using_flirt_transform(point8,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

#point B:
point9 = np.zeros([3,1])
point9[0] = 209#int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the x_coordinate of the point A: "))-1
point9[1] = 327#int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the y_coordinate of the point A: "))-1
point9[2] = 100#int(raw_input("Defining the distal talar surface directly inferior to the talar dome between A and B, please enter the z_coordinate of the point A: "))-1
point9= warp_point_using_flirt_transform(point9,nii.header, nii.header, Text_file_to_matrix(ref_matrix_talus))

ay_t= (point9-point7)*LA.norm(point8-point7) + (point8-point7)*LA.norm(point9-point7)
ay_t/= LA.norm(ay_t)

#>>> For the talus tz was defined as (∼ty×tx) and ty was defined as(tx×tz)

# The cross product of ax and ay in R^3 is a vector perpendicular to both ax and ay

az = np.cross(ax,ay_t,axis=0)
az/= LA.norm(az)

ay = np.cross(az, ax,axis=0)
ay/= LA.norm(ay)


'''####### Define the tibial coordinate system ##################
>>> Anatomically based coordinate system:
>>>  the tibial y-axis,(ty) was defined as the unit vector parallel to the tibial anterior edge in the sagittal-oblique image
'''
#point P1: inferior point from the the tibial anterior edge
point10 = np.zeros([3,1])
point10[0] = 199  #int(raw_input("please enter the x_coordinate of an inferior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
point10[1] = 397#int(raw_input("please enter the y_coordinate of an inferior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
point10[2] = 100#int(raw_input("please enter the z_coordinate of an inferior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
point10= warp_point_using_flirt_transform(point10,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

#point P2: superior point from the the tibial anterior edge

point11 = np.zeros([3,1])
point11[0] = 240 #int(raw_input("please enter the x_coordinate of a superior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
point11[1] = 509#int(raw_input("please enter the y_coordinate of a superior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
point11[2] = 100#int(raw_input("please enter the z_coordinate of a superior point from the the tibial anterior edge in the sagittal-oblique image: "))-1
point11= warp_point_using_flirt_transform(point11,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

ty = (point11-point10)/LA.norm(point11-point10)

#the temporary tibial z-axis(∼tz) was defined as the unit vector connecting the most lateral and medial tibial points

#point P3: the most external lateral tibial point
point12 = np.zeros([3,1])
point12[0] = 242  #int(raw_input("please enter the x_coordinate of the most external lateral tibial point: "))-1
point12[1] = 391#int(raw_input("please enter the y_coordinate of the most external lateral tibial point: "))-1
point12[2] = 145#int(raw_input("please enter the z_coordinate of the most external lateral tibial point: "))-1
point12= warp_point_using_flirt_transform(point12,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

#point P3: the most internal lateral tibial point
point13 = np.zeros([3,1])
point13[0] =  266 #int(raw_input("please enter the x_coordinate of the most internal lateral tibial point: "))-1
point13[1] = 391#int(raw_input("please enter the y_coordinate of the most internal lateral tibial point: "))-1
point13[2] = 59#int(raw_input("please enter the z_coordinate of the most internal lateral tibial point: "))-1
point13= warp_point_using_flirt_transform(point13,nii.header, nii.header, Text_file_to_matrix(ref_matrix_tibia))

tz_t = (point12-point13)/LA.norm(point12-point13)
tz_t/= LA.norm(tz_t)

tx = np.cross(ty, tz_t,axis=0)
tx/= LA.norm(tx)

tz = np.cross(tx, ty,axis=0)
tz/= LA.norm(tz)

'''
>>> Define origins:

Co: the calcaneal origin was defined as the insertion of the long plantar ligament

Ao: the talar origin was defined as the most inferior point on the talar dome section of the talus

To: The tibial origin was defined as the point that bisected ∼tz

'''

Co = point3
Ao = point7
To = (point12+point13)/2

### Compute the change of basis matrices

M_calcaneus = Image_to_bone_coordinate_system(image,cx,cy,cz,bone_origin=Co)
M_talus = Image_to_bone_coordinate_system(image,ax,ay,az,bone_origin=Ao)
M_tibia = Image_to_bone_coordinate_system(image,tx,ty,tz,bone_origin=To)



'''
>>> orthogonality checking
'''
'''
print("scalar product:")
print(np.dot(cx.T, cz))
print(np.dot(cx.T, cy))
print(np.dot(cy.T, cz))

print(LA.norm(cx))
print(LA.norm(cy))
print(LA.norm(cz))


print("scalar product:")
print(np.dot(ax.T, az))
print(np.dot(ax.T, ay))
print(np.dot(ay.T, az))

print(LA.norm(ax))
print(LA.norm(ay))
print(LA.norm(az))

print("scalar product:")
print(np.dot(tx.T, tz))
print(np.dot(tx.T, ty))
print(np.dot(ty.T, tz))

print(LA.norm(tx))
print(LA.norm(ty))
print(LA.norm(tz))

'''
