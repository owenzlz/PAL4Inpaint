# PA4Inpaint
[Project Page] |  [Paper](https://arxiv.org/pdf/2208.03357.pdf) | [Bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:Hg82KcKaDdkJ:scholar.google.com/&output=citation&scisdr=CgVB6GfVENSznng6iSo:AAGBfm0AAAAAYvc8kSorKdGBazl9ISGg6_ctvVJSZKcJ&scisig=AAGBfm0AAAAAYvc8kZfAWD_WLA7uXggR-vhUdJqL1ybW&scisf=4&ct=citation&cd=-1&hl=en)

<img src="https://github.com/owenzlz/PAL4Inpaint/blob/main/images/teaser.png" style="width:800px;">

**Perceptual Artifacts Localization for Inpainting**\
*European Conference on Computer Vision (ECCV), 2022, Oral Presentation*\
[Lingzhi Zhang](https://owenzlz.github.io/), [Yuqian Zhou](https://yzhouas.github.io/), [Connelly Barnes](http://www.connellybarnes.com/work/), [Sohrab Amirghodsi](https://scholar.google.com/citations?user=aFrtZOIAAAAJ&hl=en), [Zhe Lin](https://sites.google.com/site/zhelin625/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Jianbo Shi](https://www.cis.upenn.edu/~jshi/)\
University of Pennsylvania, Adobe Research/ART

Note: Due to commercial reasons, we only release our inference code in 'torchscript' format and half of the labeled datasets. 

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

## Checkpoints

- Download 'torchscript' checkpoints:
```bash
bash download_checkpoints.sh
```

## Datasets

If you want to access our labeled perceptual artifacts localization dataset, you may use the following commands to download. Running 
inference code does not require to download this. 

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

Note that labels and masks have pixel values 0 or 1. To visualize them, you can multiple the images by 255.

## Quick Usage [TODO]

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




### Citation
If you use this code for your research, please cite our paper:
```
@article{zhang2022perceptual,
  title={Perceptual Artifacts Localization for Inpainting},
  author={Zhang, Lingzhi and Zhou, Yuqian and Barnes, Connelly and Amirghodsi, Sohrab and Lin, Zhe and Shechtman, Eli and Shi, Jianbo},
  journal={arXiv preprint arXiv:2208.03357},
  year={2022}
}
```



