import os
import glob


if __name__ == '__main__':
    bones = ['calcaneus', 'talus', 'tibia']
    bone = ['fuzzy']
    for b in bone:
        print(b)
        bone_mesh = os.path.join('.', b, b + '.stl')
        subjects = glob.glob(os.path.join('.', b, 'results', '*.nii.gz'))
        for s in subjects:
            texture = 'python texture_plugger_new.py -s BSpline -is ' + bone_mesh + ' -it ' + s + ' -o ' \
                      + os.path.join('.', b, 'results', 'mesh', s.rsplit('/', 1)[1].split('.')[0] + '.ply')
            print(texture)
            os.system(texture)
