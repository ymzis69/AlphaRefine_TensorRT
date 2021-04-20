# Installation

## 1.install AlphaRefine enviroment

This document contains detailed instructions for installing the necessary dependencies for Alpha-Refine, which is 
similar to PyTracking. The instrustions have been tested on an Ubuntu 18.04 system. 

We recommend using the [install script](install.sh) if you have not already tried that. 

#### How to use [install script](install.sh) (an easy way to install AlphaRefine enviroment):

Run the installation script to install all the dependencies. You need to provide the `${conda_install_path}`
(e.g. `~/anaconda3`) and the name `${env_name}` for the created conda environment (e.g. `alpha`).

```
# install dependencies
bash install.sh ${conda_install_path} ${env_name}
#bash install.sh ~/anaconda3 alpha
conda activate alpha
python setup.py develop
```




#### Optional Setting Up for Some Experiments
```
# install pysot if you want to test AR on SiamRPNpp
cd pysot
python setup.py build_ext --inplace
cd -

# install vot-toolkit if you want to test on VOT2020
export PYTHONPATH=$PYTHONPATH:arena/VOT2020/vot
cd arena/VOT2020/vot
pip install -r requirements.txt
cd -

# install RT-MDNet if you want to test AR on RT-MDNet
cd RT_MDNet/modules/roi_align
python setup.py develop
cd -
```



### Requirements  

* Conda installation with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name alpha python=3.7
conda activate alpha
```

#### Install PyTorch  

**Since TensorRT7.0 does not support some model operations, we strongly recommend that you install TensorRT 7.1 or higher. Therefore, you will need to install the CUDA version >=10.2**.

Install PyTorch with cuda10.2.  
```bash
conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch
```

**Note:**  

- It is possible to use any PyTorch supported version of CUDA (In our experiment, we use cuda10.2+cudnn8.0.5+TensorRT7.1.3.4).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install matplotlib, pandas, tqdm, opencv, scikit-image, visdom, tikzplotlib, gdown, and tensorboad  
```bash
conda install matplotlib pandas tqdm
pip install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
```



#### Install the coco and lvis toolkits  

```bash
conda install cython
pip install pycocotools
pip install lvis
```


#### Install ninja-build for Precise ROI pooling  
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  

#### Installing pytracking as a pkg

```
python setup.py develop
```

#### Install spatial-correlation-sampler (only required for KYS tracker) 

```bash
pip install spatial-correlation-sampler
```
In case of issues, we refer to https://github.com/ClementPinard/Pytorch-Correlation-extension.  

#### Install jpeg4py  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  

#### Setup the environment  

Create the default environment setting files. 

```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  


#### 

## 2.Install TensorRT enviroment

 GO to  [NVIDIA-TENSORRT](https://developer.nvidia.com/nvidia-tensorrt-7x-download) and download a package that suits for your platform.
 We use TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar package in our experiment.

```
tar -zxf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
sudo gedit ~/.bashrc

# Add codes in your file ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your-TensorRT-lib-path>
# for example
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ymz2/TensorRT-7.1.3.4/lib

source ~/.bashrc

conda activate env_name
# conda activate alpha
cd  <your-TensorRT-path>/python
# cd /home/ymz2/TensorRT-7.1.3.4/python
# If your python version=3.7
pip install tensorrt-7.1.3.4-cp37-none-linux_x86_64.whl
cd <your-TensorRT-path>/python
# cd /home/ymz2/TensorRT-7.1.3.4/graphsurgeon
pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
```

#### install torch2trt

```
conda activate env_name
#conda activate alpha

git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

#### Verify the installation

```
python
import tensorrt
import torch2trt
```

