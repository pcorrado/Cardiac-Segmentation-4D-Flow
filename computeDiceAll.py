import os, sys, re
import nibabel as nib
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.image_utils import np_categorical_dice


testDir = '/home/pcorrado/Cardiac-DL-Segmentation-Paper/test'
numLayers = [0,4,8,12,14,15]

def compareImages(img1Path, img2Path):
    img1 = np.round(nib.load(img1Path).get_data())
    img2 = np.round(nib.load(img2Path).get_data())
    return (np_categorical_dice(img1, img2, 1), np_categorical_dice(img1, img2, 3))

if __name__ == '__main__':

    diceDict = {}
    for data in sorted(os.listdir(testDir)):
        data_dir = os.path.join(testDir, data)
        print(data_dir)
        for truthImage in sorted(os.listdir(data_dir)):
            if re.match('O\d_label_sa.nii', truthImage):
                print(os.path.join(data_dir,truthImage))
                for l in numLayers:
                    if not str(l) in diceDict:
                        diceDict[str(l)] = {}
                    if not data in diceDict[str(l)]:
                        diceDict[str(l)][data] = []
                    cnnImage = 'sa_label_{}.nii'.format(l)
                    print('Comparing {0} with {1}'.format(truthImage, cnnImage))
                    diceLV, diceRV = compareImages(os.path.join(data_dir, truthImage), os.path.join(data_dir, cnnImage))
                    diceDict[str(l)][data].append([diceLV,diceRV])

    meanDict = {}
    stdDict = {}
    for l in numLayers:
        print(l)
        arr = []
        for subj in diceDict[str(l)]:
            arr.append(np.mean(np.array(diceDict[str(l)][subj]), axis=0))
        meanDict[str(l)] = np.mean(np.array(arr), axis=0)
        stdDict[str(l)] = np.std(np.array(arr), axis=0)
        print(meanDict[str(l)])
        print(stdDict[str(l)])