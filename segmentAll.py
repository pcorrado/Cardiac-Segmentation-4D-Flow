# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

testDir = '/home/pcorrado/Cardiac-DL-Segmentation-Paper/test'
modelBasePath = '/home/pcorrado/Cardiac-DL-Segmentation-Paper/Cardiac-Segmentation-4D-Flow/TrainedModels'
modelPaths = ['model_{}_layers_frozen'.format(l) for l in [4,8,12,14,15]]
modelPaths.append('modelUnfrozen')
modelName = 'FCN_sa_level5_filter16_22333_batch20_iter10000_lr0.001'
numLayers = [4,8,12,14,15,0]

if __name__ == '__main__':

    for ii in range(len(modelPaths)):
        os.system('python3 common/deploy_network.py --data_dir {0} '
            '--model_path {1}/{2}/{3}/{3}.ckpt-10000'.format(testDir, modelBasePath, modelPaths[ii], modelName))
        for data in sorted(os.listdir(testDir)):
            data_dir = os.path.join(testDir, data)
            os.system('mv {0}/seg_sa.nii.gz {0}/sa_label_{1}.nii.gz'.format(data_dir, numLayers[ii]))


