# PA4Inpaint
[Project Page] |  [Paper] | [Bibtex]
<!-- (https://chail.github.io/latent-composition/) -->
<!-- <img src="https://github.com/owenzlz/EgoHOS/blob/main/stitch.gif" style="width:800px;"> -->

**Perceptual Artifacts Localization for Inpainting**\
*European Conference on Computer Vision (ECCV), 2022, Oral*\
[Lingzhi Zhang](https://owenzlz.github.io/), [Yuqian Zhou](https://yzhouas.github.io/), [Connelly Barnes](http://www.connellybarnes.com/work/), [Sohrab Amirghodsi](https://scholar.google.com/citations?user=aFrtZOIAAAAJ&hl=en), [Zhe Lin](https://sites.google.com/site/zhelin625/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Jianbo Shi](https://www.cis.upenn.edu/~jshi/)

Note: Due to the company policy and commercial reasons, we only release our inference code as 'torchscript' format, and will release
half of the labeled datasets. 

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Setup
- Clone this repo:
```bash
git clone https://github.com/owenzlz/PA4Inpaint
```

- Install dependencies:
```bash
pip install torch torchvision
```

- Download 'torchscript' checkpoints
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

