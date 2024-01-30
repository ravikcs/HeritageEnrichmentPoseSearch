# Heritage Enrichment Pose Search

This project work focuses on creating a visual search engine for the automatic enrichment of art collections, streamlining the retrieval process via pose based search.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Getting Started

For this work, we evaluate Yolov8's and MMpose's ViTpose-l model to estimate the human poses of artwork collections of RMFAB. For executing the code, python 3.8 or greater is required.

## Prerequisites

For YOLOv8, need to install Ultralytics package as follows:

```bash
pip install ultralytics
```
MMPose works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.8+. 

0. Download and install Anaconda
1. Create a conda environment and activate it.
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```
2. Install PyTorch
```bash
conda install pytorch torchvision -c pytorch
```
3. Install MMEngine and MMCV using MIM
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```
To develop and run mmpose directly, install it from source:
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

## Installation

To setup the project in your local directory:
```bash
git clone https://github.com/ravikcs/HeritageEnrichmentPoseSearch.git
cd HeritageEnrichmentPoseSearch
```

## Usage

For YOLOv8:
```bash
python yolov8poseart.py
```
For MMpose:
```bash
python mmposeart.py
```

## Results
Some of the successful keypoints detection by MMPose are shown below:

![1](https://github.com/ravikcs/HeritageEnrichmentPoseSearch/assets/147035848/335d11e7-9082-4860-bab6-adfc9b1c9838)
![17](https://github.com/ravikcs/HeritageEnrichmentPoseSearch/assets/147035848/3d8738dc-3bb4-44bc-b5c0-c99efd3c1461)
![22](https://github.com/ravikcs/HeritageEnrichmentPoseSearch/assets/147035848/3fcef5ea-b0d8-4bce-91ea-fdb384bf7862)

However, the model failed to detect keypoints accurately in case of drawings, battle scenes consisting of multiple people. Drawings, as hand-drawn sketches lack well-defined body structures, often disjoint body parts, displaying too much abstraction provided hurdles for the neural network, resulting in less detectable poses. However, we plan to construct a training pipeline using MMPose on such drawings providing ground truth pose keypoints to improve the results. Alternatively, using background removal techniques before pose estimation on such drawings might as well improve the results. 

![21](https://github.com/ravikcs/HeritageEnrichmentPoseSearch/assets/147035848/527eb3cc-a2db-4e0f-ad7d-267ab86a57ae)
![35](https://github.com/ravikcs/HeritageEnrichmentPoseSearch/assets/147035848/5c6fe3c3-4393-4c9e-9b33-e86d9af6d4cf)
![38](https://github.com/ravikcs/HeritageEnrichmentPoseSearch/assets/147035848/c09e2ced-ea29-4bc3-b052-144cf0cf2571)
![45](https://github.com/ravikcs/HeritageEnrichmentPoseSearch/assets/147035848/56cf3719-b7d0-45bd-8031-7fa74f300724)
