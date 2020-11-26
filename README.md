# Channel-wise Distillation for Semantic Segmentation

## Introduction

This repository contains the PyTorch test implementation of:  Channel-wise Distillation for Semantic Segmentation

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


#### 1. Inference with the evaluation dataset

```bash
python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.02_test-73.86.pth --gpu 0 --type val  --figsavepath 75.02val

python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.90_test-74.58.pth --gpu 0 --type val  --figsavepath 75.90val
```

#### 2. Inference on the test dataset

```bash
python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.02_test-73.86.pth --gpu 0 --type test --figsavepath 73.86test

python valandtest.py --data-dir path/to/CITYSCAPES/data/cityscapes  --restore-from ckpt/new_rn18-cityscape_singleAndWhole_val-75.90_test-74.58.pth --gpu 0 --type test --figsavepath 74.58test
```


| Model | Average | roda | sidewalk | building|	wall | fence | pole | trafficlight | trafficsign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| IoU | 73.86 | 97.84 | 81.01 | 91.55 | 48.63 | 53.18 | 61.14 | 70.21 | 74.20 | 92.93 | 70.91 | 94.84 | 83.11 | 62.39 | 94.74 | 54.12 | 66.80 | 70.91 | 61.60 | 73.27 |
| IoU | 74.58 | 97.78 | 80.56 | 91.45 | 52.78 | 52.91 | 59.90 | 70.50 | 73.13 | 92.54 | 70.70 | 94.57 | 82.25 | 63.51 | 94.76 | 59.31 | 73.68 | 73.00 | 61.54 | 72.12 |



## Citation

Please consider citing this work if it helps your research:

```

@inproceedings{shu2020cwd,
  title={Channel-wise Distillation for Semantic Segmentation},
  author={Shu, Changyong and Liu, Yifan and Gao, Jianfei and Xu, Lin and Shen, Chunhua},
}

```

