# Channel-wise Distillation for Semantic Segmentation

## Introduction

This repository contains the PyTorch implementation of: 

Channel-wise Distillation for Semantic Segmentation

## Requirements

All the codes are tested in the following environment:

* Linux (tested on Ubuntu 16.04 / CentOS 7.6)
* Python 3.6.2
* PyTorch 0.4.1
* Single TITAN Xp GPU

## Installation

* Install PyTorch: ` conda install pytorch=0.4.1 cuda90 torchvision -c pytorch `
* Install other dependences: ` pip install opencv-python scipy `
* Install InPlace-ABN:
```bash
cd libs
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

## Dataset & Models

* Dataset: [[Cityscapes]](https://www.cityscapes-dataset.com/)

* After distillation: PSPNet (ResNet-18) 
 rn18-cityscape_singleAndWhole_val-75.05_test-73.86.pth [[Google Drive]](https://drive.google.com/file/d/1eLOslSm1Clif_PJFTedbmG9fdhhqPSAe/view?usp=sharing)
 rn18-cityscape_singleAndWhole_val-75.90_test-74.58.pth [[Google Drive]](https://drive.google.com/file/d/1IWGQvoP8OMcRysHPMPmXAjWi8k7IW3ZZ/view?usp=sharing)

Please create a new folder `ckpt` and move all downloaded models to it.

## Usage


(1)  Inference with evaluation

(2)  Inference with test (run via the following, then submit the result to https://www.cityscapes-dataset.com/)

#### 1. Inference with evaluation

```bash

python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.02_test-73.86.pth --gpu 0 --type val  --figsavepath 75.02val
python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.90_test-74.58.pth --gpu 0 --type val  --figsavepath 75.90val

```  

#### 2. Inference only
  

```bash

python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.02_test-73.86.pth --gpu 0 --type test --figsavepath 73.86test
python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.90_test-74.58.pth --gpu 0 --type test --figsavepath 74.58test

```

## Citation

Please consider citing this work if it helps your research:

```

@inproceedings{wang2020ifvd,
  title={Channel-wise Distillation for Semantic Segmentation},
  author={Shu, Changyong and Liu, Yifan and Gao, Jianfei and Xu, Lin and Shen, Chunhua},
}

```

