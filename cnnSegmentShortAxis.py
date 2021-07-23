import sys
import os
import time
import random
import numpy as np
import nibabel as nib
import tensorflow as tf
import math
from scipy.ndimage import zoom
sys.path.insert(1, '/export/home/pcorrado/CODE/')

print(sys.path)
from ukbb_cardiac.common.network import build_FCN
from ukbb_cardiac.common.image_utils import tf_categorical_accuracy, tf_categorical_dice
from ukbb_cardiac.common.image_utils import crop_image, rescale_intensity, data_augmenter
from ukbb_cardiac.common import deploy_network

if os.path.exists("sa.nii.gz"):
    nim = nib.load("sa.nii.gz")
elif os.path.exists("sa.nii"):
    nim = nib.load("sa.nii")
orig_image = nim.get_data()
(X1,Y1,Z1,T1) = orig_image.shape

if X1 != Y1:
    print("Image is not square, exiting.")
else:
    print('Pre-zoom image shape')
    print(orig_image.shape)
    image = zoom(orig_image,(192.0/X1,192.0/Y1,1,1),order=1)
    print('Post-zoom image shape')
    print(image.shape)
    os.system('CUDA_VISIBLE_DEVICES=1')
    model_path = os.path.join(os.path.dirname(__file__), "modelFT/FCN_sa_level5_filter16_22333_batch20_iter10000_lr0.001/FCN_sa_level5_filter16_22333_batch20_iter10000_lr0.001.ckpt")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(model_path))
        saver.restore(sess, '{0}'.format(model_path))

        print('Start deployment on the data set ...')
        start_time = time.time()

        X, Y, Z, T = image.shape

        print('  Segmenting full sequence ...')
        start_seg_time = time.time()

        # Intensity rescaling
        image = rescale_intensity(image, (1, 99))

        # Prediction (segmentation)
        pred = np.zeros(image.shape)

        # Pad the image size to be a factor of 16 so that the
        # downsample and upsample procedures in the network will
        # result in the same image size at each resolution level.
        X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
        x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
        x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

        # Process each time frame
        for t in range(T):
            # Transpose the shape to NXYC
            image_fr = image[:, :, :, t]
            image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
            image_fr = np.expand_dims(image_fr, axis=-1)

            # Evaluate the network
            prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                        feed_dict={'image:0': image_fr, 'training:0': False})

            # Transpose and crop segmentation to recover the original size
            pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
            pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
            pred[:, :, :, t] = pred_fr

        seg_time = time.time() - start_seg_time
        print('  Segmentation time = {:3f}s'.format(seg_time))

        pred = zoom(pred,(X1/X,Y1/Y,1,1),order=0)
        rvMask = pred==3
        pred[rvMask]=-1

        # Save the segmentation
        print('  Saving segmentation ...')
        nim2 = nib.Nifti1Image(pred, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        seg_name = './seg_sa.nii.gz'
        nib.save(nim2, seg_name)
