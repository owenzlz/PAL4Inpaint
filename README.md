# PA4Inpaint
[Project Page] |  [Paper] | [Bibtex]
<!-- (https://chail.github.io/latent-composition/) -->
<!-- <img src="https://github.com/owenzlz/EgoHOS/blob/main/stitch.gif" style="width:800px;"> -->

**Perceptual Artifacts Localization for Inpainting**\
*European Conference on Computer Vision (ECCV), 2022, Oral Presentation*\
[Lingzhi Zhang](https://owenzlz.github.io/), [Yuqian Zhou](https://yzhouas.github.io/), [Connelly Barnes](http://www.connellybarnes.com/work/), [Sohrab Amirghodsi](https://scholar.google.com/citations?user=aFrtZOIAAAAJ&hl=en), [Zhe Lin](https://sites.google.com/site/zhelin625/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Jianbo Shi](https://www.cis.upenn.edu/~jshi/)


## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

**Table of Contents:**<br>
1. [Setup](#setup) - download pretrained models and resources
2. [Datasets](#datasets) - download our egocentric hand-object segmentation datasets
3. [Quick Usage](#pretrained) - quickstart with pretrained models<br>


## Setup
- Clone this repo:
```bash
git clone https://github.com/owenzlz/PA4Inpaint
```

- Install dependencies:
```bash
pip install torch torchvision
```

- Download resources:
	- we provide a script for downloading the pretrained checkpoints. 
```bash
bash download_checkpoints.sh
```

## Datasets
- Download our dataset from GDrive links (), or use the following command line.
```bash
gdown ...
```

After downloading, the dataset is structured as follows: 
```bash
- [egohos dataset root]
    |- train
        |- image
        |- label
    |- val 
        |- image
        |- label
    |- test
        |- image
        |- label
```

## Quick Usage

Run the following command for inference. 

- Predict left and right hands
```bash
bash ...
```

