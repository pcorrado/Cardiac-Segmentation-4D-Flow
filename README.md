## Overview

This code is based on a toolbox developed by Wenjia Bai (https://github.com/baiwenjia/ukbb_cardiac) for processing cardiovascular magnetic resonance (CMR) images.

I have modified the code for "fine-tuning", or loading in the weights from training on a large data set (UK BioBank) and training the weights for a small number of iterations on data acquired at my local institution. There are also additional scripts for registering bSSFP images to 4D flow images and for computing kinetic energy and flow components from masked 4D flow MRI images. The registration scripts use the ANTs toolbox (http://stnava.github.io/ANTs/) and the analysis scripts run in Matlab.


## Segmentation Toolbox Installation

The segmentation toolbox is developed using Python 3.

The toolbox depends on some external libraries which need to be installed, including:

* tensorflow for deep learning;
* numpy and scipy for numerical computation;
* pandas and python-dateutil for handling spreadsheet;
* pydicom, SimpleITK for handling dicom images
* nibabel for reading and writing nifti images;
* opencv-python for transforming images in data augmentation.

The most convenient way to install these libraries is to use pip3 (or pip for Python 2) by running this command in the terminal:
```
pip3 install tensorflow-gpu numpy scipy pandas python-dateutil pydicom SimpleITK nibabel opencv-python
```

To use, please add the github repository directory to your $PYTHONPATH environment, so that the ukbb_cardiac module can be imported and cross-referenced in its code. If you are using Linux, you can run this command:
```
export PYTHONPATH=YOUR_GIT_REPOSITORY_PATH:"${PYTHONPATH}"
```

There is one parameter in the script, *CUDA_VISIBLE_DEVICES*, which controls which GPU device to use on your machine. Currently, I set it to 0, which means the first GPU on your machine.
