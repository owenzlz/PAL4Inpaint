# PA4Inpaint
[Project Page] |  [Paper] | [Bibtex]
<!-- (https://chail.github.io/latent-composition/) -->

<img src="https://github.com/owenzlz/PA4Inpaint/blob/main/images/teaser.png" style="width:800px;">

**Perceptual Artifacts Localization for Inpainting**\
*European Conference on Computer Vision (ECCV), 2022, Oral Presentation*\
[Lingzhi Zhang](https://owenzlz.github.io/), [Yuqian Zhou](https://yzhouas.github.io/), [Connelly Barnes](http://www.connellybarnes.com/work/), [Sohrab Amirghodsi](https://scholar.google.com/citations?user=aFrtZOIAAAAJ&hl=en), [Zhe Lin](https://sites.google.com/site/zhelin625/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Jianbo Shi](https://www.cis.upenn.edu/~jshi/)

Note: Due to commercial reasons, we only release our inference code as 'torchscript' format and half of the labeled datasets. 

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Setup
- Clone this repo:
```bash
git clone https://github.com/owenzlz/PA4Inpaint
```

- Install dependencies (Since we use torchscript, we can bypass the mmsegmentation packages.):
```bash
pip install torch torchvision
```

- Download 'torchscript' checkpoints:
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

Note that labels and masks have pixel values 0 or 1. To visualize them clearly, you can multiple the images by 255. 

## Quick Usage

Run the following command for inference. 

- Inference on a single image:
```bash
python pa4inpaint.py \
--img_file ./demo/images/xxx.jpg \
--result_file ./demo/results/yyy.png
```

- Inference on a batch of images:
```bash
python pa4inpaint.py \
--img_dir ./demo/images \
--result_dir ./demo/results
```

<img src="https://github.com/owenzlz/PA4Inpaint/blob/main/images/seg_results.png" style="width:800px;">


