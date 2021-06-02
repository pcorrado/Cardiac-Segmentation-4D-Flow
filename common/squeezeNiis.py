import numpy as np
import nibabel as nib
import os

def squeezeNii(root, file):
    img = nib.load(root + '/' + file)
    newFile = file.replace('.nii','_temp.nii')
    nib.save(nib.Nifti1Image(np.squeeze(img.dataobj),img.affine), root+'/'+newFile)


# This is to get the directory that the program
# is currently running in.
dir_path = '/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2020_CARDIAC_DL_SEGMENTATION_CORRADO/test'

for root, dirs, files in os.walk(dir_path):
    print(root)
    for file in files:
        if file.endswith('.nii'):
            squeezeNii(root, str(file))
            os.remove(root + '/' + str(file))
            os.rename(root + '/' + str(file).replace('.nii','_temp.nii'), root + '/' + str(file))
