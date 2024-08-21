# SG-MIM: Structured Knowledge Guided Masked Image Modeling

## Overview

SG-MIM (Structured Knowledge Guided Masked Image Modeling) is a framework that enhances Masked Image Modeling (MIM) by integrating structured knowledge during the pre-training process. This approach aims to improve the performance of dense prediction tasks, particularly in depth estimation and semantic segmentation, by effectively encoding spatially structured information.

## Features

- **Relational Guidance Framework**: A lightweight module that encodes structured knowledge separately from the main image encoder, providing feature-level guidance.
- **Selective Masking Strategy**: A technique that balances learning difficulty by focusing on visible image regions, enhancing the model's ability to learn fine-grained features.
- **Efficient Pre-training**: SG-MIM outperforms existing MIM models across various tasks with fewer epochs and reduced computational resources.
- **Application in Dense Prediction**: Demonstrates superior performance in tasks like monocular depth estimation and semantic segmentation.

## Installation

To use SG-MIM, you need to have the following dependencies installed:

- Python 3.8
- PyTorch 1.12.0
- CUDA 11.3
- Other dependencies listed in `sgmim.yaml`

To install the dependencies, run:

```bash
conda env create --file environment.yaml
conda activate sgmim
```
## Pre-training
For pre-training, SG-MIM utilizes Swin-Base, Swinv2-Base, and ViT-Base backbones.

To run the training code,
```bash
python -m torch.distributed.launch --nproc_per_node {gpu_num} main_sgmim.py --cfg ./configs/{config_file} --batch-size  128 --data-path {imagenet_train_path} --depth-data-path {structured_data_path} --file-name-path ImageNet_train.txt 
```

## Data
You can get ImageNet Dataset [official websites](https://image-net.org/download.php), and structured dataset can get [here](https://github.com/EPFL-VILAB/MultiMAE/blob/main/SETUP.md).