# Copyright 2017, Wenjia Bai. All Rights Reserved.
# Modified in 2020 by Philip Corrado.
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
import shutil


if __name__ == '__main__':
    # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    testDir = '/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2020_CARDIAC_DL_SEGMENTATION_CORRADO/test/'
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir {1} '
           '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES,testDir))
    for data in sorted(os.listdir(testDir)):
        data_dir = os.path.join(testDir, data)
        print('Next dir to sort: {0}'.format(data_dir))
        os.system('mkdir {0}/ukbb && mv {0}/*.nii.gz {0}/ukbb'.format(data_dir))


    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir {1} '
          '--model_path modelFT/FCN_sa_level5_filter16_22333_batch20_iter3000_lr0.001/FCN_sa_level5_filter16_22333_batch20_iter3000_lr0.001.ckpt-3000 '.format(CUDA_VISIBLE_DEVICES,testDir))
    for data in sorted(os.listdir(testDir)):
        data_dir = os.path.join(testDir, data)
        print('Next dir to sort: {0}'.format(data_dir))
        os.system('mkdir {0}/ft && mv {0}/*.nii.gz {0}/ft'.format(data_dir))

    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir {1} '
           '--model_path model/FCN_sa_level5_filter16_22333_batch20_iter3000_lr0.001/FCN_sa_level5_filter16_22333_batch20_iter3000_lr0.001.ckpt-3000'.format(CUDA_VISIBLE_DEVICES,testDir))
    for data in sorted(os.listdir(testDir)):
        data_dir = os.path.join(testDir, data)
        print('Next dir to sort: {0}'.format(data_dir))
        os.system('mkdir {0}/uw && mv {0}/*.nii.gz {0}/uw'.format(data_dir))
