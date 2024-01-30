# Heritage Enrichment Pose Search

This project work focuses on creating a visual search engine for the automatic enrichment of art collections, streamlining the retrieval process via pose based search.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

For this work, we evaluate Yolov8's and MMpose's ViTpose-l model to estimate the human poses of artwork collections of RMFAB. For executing the code, python 3.8 or greater, 

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

Provide step-by-step instructions on how to install and set up the project.

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## Usage

Explain how to use the code. Provide examples or command snippets to demonstrate its functionality.

```bash
python main.py --option value
```

## Results
Some of the successful keypoints detection by MMPose are shown below:

However, the model failed to detect keypoints accurately in case of drawings, battle scenes consisting of multiple people. Drawings, as hand-drawn sketches lack well-defined body structures, often disjoint body parts, displaying too much abstraction provided hurdles for the neural network, resulting in less detectable poses. However, we plan to construct a training pipeline using MMPose on such drawings providing ground truth pose keypoints to improve the results. Alternatively, using background removal techniques before pose estimation on such drawings might as well improve the results. 


## Contributing

If you'd like to contribute to the project, provide guidelines for how others can do so. Include information about how to submit issues, propose new features, or submit pull requests.

## License

This project is licensed under the [License Name] - see the [LICENSE.md](LICENSE.md) file for details. (Replace `[License Name]` with the actual license name and provide the license file if applicable.)

Feel free to customize this template according to the specifics of your project.
