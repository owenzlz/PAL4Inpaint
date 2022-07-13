# PA4Inpaint
[Project Page] |  [Paper] | [Bibtex]
<!-- (https://chail.github.io/latent-composition/) -->

<img src="https://github.com/owenzlz/PA4Inpaint/blob/main/images/teaser.png" style="width:800px;">

**Perceptual Artifacts Localization for Inpainting**\
*European Conference on Computer Vision (ECCV), 2022, Oral Presentation*\
[Lingzhi Zhang](https://owenzlz.github.io/), [Yuqian Zhou](https://yzhouas.github.io/), [Connelly Barnes](http://www.connellybarnes.com/work/), [Sohrab Amirghodsi](https://scholar.google.com/citations?user=aFrtZOIAAAAJ&hl=en), [Zhe Lin](https://sites.google.com/site/zhelin625/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Jianbo Shi](https://www.cis.upenn.edu/~jshi/)

Note: Due to some commercial reasons, we only release our inference code as 'torchscript' format, and will release
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

<img src="https://github.com/owenzlz/PA4Inpaint/blob/main/images/user_labels.png" style="width:800px;">

- Download our datasets
```bash
bash download_datasets.sh
```

After downloading, the dataset is structured as follows: 
```bash
- [perceptual artifacts dataset root]
    |- trainset
        |- images
        |- masks
        |- labels
    |- valset
        |- images
        |- masks
        |- labels
    |- testset
        |- images
        |- masks
        |- labels
```

## Quick Usage

Run the following command for inference. 

- Predict left and right hands
```bash
bash ...
```

<img src="https://github.com/owenzlz/PA4Inpaint/blob/main/images/seg_results.png" style="width:800px;">