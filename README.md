# TRFS
A Tri-Branch Feature Rectification and Fusion Network for RGB-X-Y Multimodal Semantic Segmentation
##  Introduction

Multimodal semantic segmentation plays a crucial role in complex scene understanding, particularly in demanding applications such as infrastructure inspection and subway tunnel disease detection. While traditional dual-branch networks have significantly advanced RGB-X fusion, they frequently struggle with cross-modal spatial misalignment and the inherent noise present in heterogeneous sensor data.

To address these critical challenges, we introduce **TRFS**, a novel **Tri-branch Multimodal Semantic Segmentation Network**. Diverging from conventional dual-stream architectures, TRFS introduces a specialized third branch to explicitly handle modality alignment and suppress noise propagation during the feature fusion process. 

Our framework is highly optimized for the robust fusion of **RGB images and LiDAR point cloud data**, effectively leveraging the rich photometric textures from RGB and the precise geometric structure from LiDAR to produce highly accurate segmentation maps.
## Installation

1. Requirements

* Python 3.8+
* PyTorch 1.7.0 or higher
* CUDA 10.2 or higher

We have tested the following versions of OS and softwares:

* OS: Ubuntu 18.04.6 LTS
* CUDA: 12.1
* PyTorch 2.3.1
* Python 3.8.11

2. Install all dependencies. Install pytorch, cuda and cudnn, then install other dependencies via:

```bash
pip install -r requirements.txt
```
## Datasets

The data folder is structured as:

```text
<DATASET_ROOT>/
├── cityscapes-DDC/
│   ├── RGB/
│   ├── leftImg8bit/
│   ├── gtFine/
│   ├── depth/
│   ├── train.txt
│   └── test.txt
├── STSD/
│   ├── RGB/
│   ├── Channel_H/
│   ├── Channel_Z/
│   ├── labels/
│   ├── train.txt
│   └── test.txt
├── MCubeS/
│   ├── RGB/
│   ├── NIR/
│   ├── DoLP/
│   ├── labels/
│   ├── train.txt
│   └── test.txt
└── MFNet/
    ├── RGB/
    ├── NIR/
    ├── Thermal/
    ├── labels/
    ├── train.txt
    └── test.txt
```
train.txt contains the names of items in training set, e.g.:
```
<name1>
<name2>
...
```
To intuitively illustrate the heterogeneous characteristics and data distributions of the aforementioned multimodal datasets, we provide a visualization of typical samples from each dataset below. The figures showcase the strictly aligned cross-modal results spanning standard RGB images and various heterogeneous sensor modalities—such as Depth/Disparity maps, Channel H/Z, Near-Infrared (NIR), Degree of Linear Polarization (DoLP), and Thermal infrared—alongside their corresponding semantic segmentation ground truths.   
<p align="center">
  <img src="docs/dataset_samples.png" alt="Multimodal Dataset Samples" width="100%">
</p>

> **Figure:** *The datasets used are MFNet, MCubeS, Cityscapes-DDC, and STSD. The X-modal input data consists of thermal infrared, near-infrared, binocular stereo matching disparity maps, and local H-axis channel images of point cloud projections, in that order. The Y-modal input data consists of linear polarization, depth maps, and local Z-axis channel images of point cloud projections, in that order.*
For preparation of the datasets, please refer to their official websites or our provided mirrors:

* [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [STSD](https://github.com/lichking2017/STSD)
* [MCubeS](https://multimodal-material-segmentation.github.io/)
## Models

Currently, we provide code of:

<p align="center">
  <img src="docs/TB-CM-FRM.png" alt="TB-CMFRM Module" width="48%">
  &nbsp; &nbsp; 
  <img src="docs/TB-FFM.png" alt="TB-FFM Module" width="48%">
</p>

> **Figure:** *Detailed architecture of our proposed modules. **Left:** The Tri-Branch Cross-Modal Feature Rectification Module (TB-CMFRM). **Right:** The Tri-Branch Feature Fusion Module (TB-FFM).*
## Train

1. Pretrain weights:
   
   Download the pretrained segformer here [pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5).
   ```
    checkpoints/pretrained/segformer
    ├── mit_b2.pth
    └── mit_b4.pth
   ```

2. Config
   
   Edit config file in `configs.py`, including dataset and network settings.

3. Run single GPU  training:
 
 ```bash
   python train.py --device 0
 ```
4.  Run multi GPU distributed training:

```bash
$ CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers" train.py
```
## Evaluation
Run the evaluation by:
```
CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py -d="Device ID" -e="epoch number or range"
```
If you want to use multi GPUs please specify multiple Device IDs (0,1,2...).
##  visualization
## Visualization

Here, we provide comprehensive visual comparisons of our semantic segmentation results across different challenging scenarios. The outputs demonstrate how our network effectively fuses RGB and LiDAR/heterogeneous data to achieve precise boundary delineation and noise suppression.

## Visualization

Here, we provide comprehensive visual comparisons of our semantic segmentation results across different challenging scenarios. The outputs demonstrate how our network effectively fuses RGB and LiDAR/heterogeneous data to achieve precise boundary delineation and noise suppression.

<p align="center">
  <img src="docs/mfnet.png" alt="Visualization 1" width="100%">
</p>

<br>

<p align="center">
  <img src="docs/mcubes.png" alt="Visualization 2" width="100%">
</p>

<br>

<p align="center">
  <img src="docs/stsd.png" alt="Visualization 3" width="100%">
</p>

<br>

<p align="center">
  <img src="docs/cityscapes.png" alt="Visualization 4" width="100%">
</p>
