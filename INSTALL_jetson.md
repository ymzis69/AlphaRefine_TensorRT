# Installation

This document contains detailed instructions for installing the necessary dependencies for Alpha-Refine, which is similar to PyTracking. The instrustions have been tested on an jetson xavier nx with ubuntu18.04 system, cuda10.2 and TensorRT.


#### Optional Setting Up for Some Experiments
```
# install pysot if you want to test AR on SiamRPNpp
cd pysot
python3 setup.py build_ext --inplace
cd -

# install vot-toolkit if you want to test on VOT2020
export PYTHONPATH=$PYTHONPATH:arena/VOT2020/vot
cd arena/VOT2020/vot
pip3 install -r requirements.txt
cd -

# Some bugs while execute this command
# install RT-MDNet if you want to test AR on RT-MDNet
# cd RT_MDNet/modules/roi_align
# sudo python3 setup.py develop
# cd -
```



## Step-by-step instructions  
In our experiment, we use python3.6.9+pytorch1.6.0+torchision0.6.0.

#### install python enviroment

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git cmake python3-dev
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools
```

#### install pytorch1.6.0

```bash
sudo apt-get install libopenblas-base libopenmpi-dev
sudo pip3 install mpi4py
sudo pip3 install Cython
sudo pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
```

#### install torchvision0.6.0

```bash
sudo apt-get install libjpeg-dev zlib1g-dev
git clone --branch  v0.6.0 https://github.com/pytorch/vision torchvision
cd torchvision
sudo python3 setup.py install
```

#### Install matplotlib, pandas, tqdm, opencv, scikit-image, visdom, tikzplotlib, gdown, and tensorboad  
```bash
pip3 install matplotlib pandas tqdm
pip3 install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
```


#### Install the coco and lvis toolkits  
```bash
pip3 install pycocotools
pip3 install lvis
```


#### Install ninja-build for Precise ROI pooling  
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  

#### Installing pytracking as a pkg

```
sudo python3 setup.py develop
```


#### Install spatial-correlation-sampler (only required for KYS tracker) 
```bash
pip3 install spatial-correlation-sampler
```
In case of issues, we refer to https://github.com/ClementPinard/Pytorch-Correlation-extension.  

#### Install jpeg4py  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip3 install jpeg4py 
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  

#### Setup the environment  

Create the default environment setting files. 

```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python3 -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python3 -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  

####

TensorRT is already installed on the Jetson device, you just need to install torch2trt.

#### Install torch2trt

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python3 setup.py install
```

#### Verify the tensorrt and torch2trt

```
python3
import tensorrt
import torch2trt
```



### Docker image:

if you want to use docker,you can use command: 

```
sudo docker pull registry.cn-hangzhou.aliyuncs.com/zxh98/alpharefine_jetson_trt
```

Note：

1.The code and test dataset needs to be mounted under the `/home` directory;

2.There is no `scikit-image` library in the environment.