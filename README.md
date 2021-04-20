# Alpha-Refine_TensorRT

Notes:

​	1.This repo only includes tensorRT version of AlphaRefine module, not including other base trackers.

​	2.This repo only support test(maybe train is no problem), but we suggest that if you want to train a network model, you should use the base link.

##### Base: Alpha-Refine(https://github.com/MasterBin-IIAU/AlphaRefine)



## Getting Start

#### Setting Up

```bash
git clone https://github.com/ymzis69/AlphaRefine_TensorRT.git
cd AlphaRefine
```



#### Install enviroment

Differnet from the base link, you need to install [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-7x-download) and [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt).(TensorRT version>=7.1)

If you use this repo in NVIDIA embedded products(such as jetson xavier nx), you can follow this [install_jetson.md]( install_jetson.md).

if you want to use TensorRT in Desktop computer,you can follow this [install.md](install.md).



#### Download AlphaRefine Models
We recommend download the **pytorch** model into `ltr/checkpoints/ltr/SEx_beta`, If you want to modify the test configuration, I suggest you take a look at these two files:`arena/LaSOT/common_path.py` and `arena/LaSOT/common_path_siamrpn.py`(You must modify the path in   `arena/LaSOT/common_path.py`    to make sure it run properly).

The TensorRT model is converted and tested in jetson xavier nx. **If you want to use TensorRT model in your device, we suggest that you should download the pytorch model and convert the TensorRT model in your device.**

on UAV123 dataset, The results of AlphaRefine:

|                       | success | precision |   speed   |                                                              |
| :-------------------: | :-----: | :-------: | :-------: | :----------------------------------------------------------: |
|      AR-alexnet       |  0.503  |   0.644   |   12fps   | [model](https://drive.google.com/file/d/1TgDaUi87g0kL_MpPdeZckuKnJ8a2Ojpj/view?usp=sharing) |
|      AR-resnet34      |  0.541  |   0.688   |   10fps   | [model](https://drive.google.com/file/d/1OJopZnpSh7Rqc20RZjfY-8r_ms1_RxFY/view?usp=sharing) |
|   AR-efficientnetb0   |  0.561  |   0.715   |   15fps   | [model](https://drive.google.com/file/d/1WW6S8SkbWnxqxAaRhAo5JmjLjTQNnRbM/view?usp=sharing) |
|     AR-mobilenet      |  0.527  |   0.671   |   18fps   | [model](https://drive.google.com/file/d/1aBKQBIOCtDm1dXJyB1DyIf0FKVzYqHSu/view?usp=sharing) |
|    AR-alexnet-trt     |  0.506  |   0.648   | **38fps** | [model](https://drive.google.com/drive/folders/1veVFnVVjz9So5KKaYrRl4mtJ9HidJQyR?usp=sharing) |
|    AR-resnet34-trt    |  0.549  |   0.697   | **39fps** | [model](https://drive.google.com/drive/folders/1xBwyQMfECxuyfOT2QnVqfOfroFyzx7t2?usp=sharing) |
| AR-efficientnetb0-trt |  0.561  |   0.715   | **40fps** | [model](https://drive.google.com/drive/folders/1eSbNlzl-xVG8SWjiKF8RbXtOZPwPx55M?usp=sharing) |
|   AR-mobilenet-trt    |  0.521  |   0.664   | **49fps** | [model](https://drive.google.com/drive/folders/17a1AaoFA3AppGu7Ls_g-BxiJTK7aJzzv?usp=sharing) |



#### Convert TensorRT Models

Run the following command to convert TensorRT model, support alexnet、resnet34、efficientnetb0 and mobilenet(You can modify the configuration in `arena/LaSOT/common_path.py` to determine convert which model).

Take alexnet as an example:

```
refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet-alex_ep0040-a.pth.tar'                                   # alexnet
# refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet_ep0040-c.pth.tar'                                         # resnet34
# refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet-efb0_ep0040-a.pth.tar'                              # efficientnetb0
# refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet-mbv2_ep0040-c.pth.tar'                            # mobilenetv2
```

Run the following command to convert the model:

```
python ./arena/LaSOT/run_RF_RF.py --tracker_name siamrpn_r50_l234_dwxcorr --dataset UAV123 --convert_trt True
```

If you want to run navie AlphaRefine, you can set the `conver_trt` parameter to False:

```
python ./arena/LaSOT/run_RF_RF.py --tracker_name siamrpn_r50_l234_dwxcorr --dataset UAV123 --convert_trt False
```



If you have succeed convert the TensorRT model, the location of the TensorRT model is as follows(take alexnet as an example):

```
├── arena
├── ltr
├── pysot
├── pytracking
├── RT_MDNet
├── trt_models
│   ├── trt_model_RF_RF_alex  
│   │   ├── backbone_alex_256_trt.pth
│   │   ├── channel_attention_trt.pth
│   │   ├── corner_head_trt.pth
│   │   ├── feat_adjust_0_trt.pth
│   │   ├── feat_adjust_1_trt.pth
│   │   ├── feat_adjust_2_trt.pth
│   │   ├── mask_head_trt.pth
│   │   ├── post_corr_trt.pth
│   ├── trt_model_RF_RF_r34
│   ├── trt_model_RF_RF_efb0
│   ├── trt_model_RF_RF_mbv2
```



#### Run TensorRT Models

run the following command to test the TensorRT model: alexnet(1)、resnet34(2)、efficientnetb0(3) and mobilenet(4).

```
python ./arena/LaSOT/run_RF_RF_trt.py --tracker_name siamrpn_r50_l234_dwxcorr --dataset UAV123 --tensorrt_model 1
                                                                                                                                                                                                                                                         2
                                                                                                                                                                                                                                                         3                                                                                               
                                                                                                                                                                                                                                                         4
```

The results will be saved in the folder `analysis`. 



#### Optional test

Test AR+ BaseTracker in this repository, you should download base models firstly.

DiMP50, DiMPsuper, ATOM, ECO are trackers from [PyTracking](pytracking).

The base tracker models trained using PyTracking can be download from [model zoo](https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md), download them into `pytracking/networks` 

Or you can run the following script to download the models.

```
# "****************** Downloading networks ******************"
mkdir pytracking/networks

# "****************** DiMP Network ******************"
gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth
gdown https://drive.google.com/uc\?id\=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk -O pytracking/networks/dimp18.pth
gdown https://drive.google.com/open?id=1qDptswis2FxihLRYLVRGDvx6aUoAVVLv -O pytracking/networks/super_dimp.pth

# "****************** ATOM Network ******************"
gdown https://drive.google.com/uc\?id\=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU -O pytracking/networks/atom_default.pth

# "****************** ECO Network ******************"
gdown https://drive.google.com/uc\?id\=1aWC4waLv_te-BULoy0k-n_zS-ONms21S -O pytracking/networks/resnet18_vggmconv1.pth
```

##### Other Base Trackers

Please refer to [pysot/README.md](pysot/README.md) for establishing SiamRPN++ and [RT_MDNet/README.md](RT_MDNet/README.md) for establishing RTMDNet.




## Alpha-Refine is Based on PyTracking Code Base
PyTracking is a general python framework for visual object tracking and video object segmentation,
based on **PyTorch**.


### Base Trackers
The toolkit contains the implementation of the following trackers.  

##### PrDiMP
**[[Paper]](https://arxiv.org/pdf/2003.12565)  [[Raw results]](MODEL_ZOO.md#Raw-Results)
  [[Models]](MODEL_ZOO.md#Models)  [[Training Code]](./ltr/README.md#PrDiMP)  [[Tracker Code]](./pytracking/README.md#DiMP)**
    

##### DiMP
**[[Paper]](https://arxiv.org/pdf/1904.07220)  [[Raw results]](MODEL_ZOO.md#Raw-Results)
  [[Models]](MODEL_ZOO.md#Models)  [[Training Code]](./ltr/README.md#DiMP)  [[Tracker Code]](./pytracking/README.md#DiMP)**
    

##### ATOM
**[[Paper]](https://arxiv.org/pdf/1811.07628)  [[Raw results]](MODEL_ZOO.md#Raw-Results)
  [[Models]](MODEL_ZOO.md#Models)  [[Training Code]](./ltr/README.md#ATOM)  [[Tracker Code]](./pytracking/README.md#ATOM)**  


##### ECO
**[[Paper]](https://arxiv.org/pdf/1611.09224.pdf)  [[Models]](https://drive.google.com/open?id=1aWC4waLv_te-BULoy0k-n_zS-ONms21S)  [[Tracker Code]](./pytracking/README.md#ECO)**  


## Acknowledgments
* This repo is based on [Pytracking](https://github.com/visionml/pytracking.git) which is an exellent work.
* Thansk for [pysot](https://github.com/STVIR/pysot) and [RTMDNet](https://github.com/IlchaeJung/RT-MDNet) from which we
 we borrow the code as base trackers.

