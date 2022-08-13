from skimage.io import imsave
from PIL import Image
import numpy as np 
import random
import glob
import pdb
import os 

alpha = 0.4
n_sample = 5
img_dir = './datasets/train_subset/images'
msk_dir = './datasets/train_subset/masks'
lbl_dir = './datasets/train_subset/labels'
vis_dir = './vis_labels'; os.makedirs(vis_dir, exist_ok = True)

img_file_list = random.sample(glob.glob(img_dir + '/*'), n_sample)

for img_file in img_file_list:

    fname = os.path.basename(img_file)
    img = np.array(Image.open(img_file).convert('RGB'))
    msk = np.array(Image.open(os.path.join(msk_dir, fname)).convert('RGB')); msk[msk > 0] = 1
    lbl = np.array(Image.open(os.path.join(lbl_dir, fname)).convert('RGB')); lbl[lbl > 0] = 1
    blue = np.zeros((img.shape)); blue[:,:,2] = 255
    pink = np.zeros((img.shape)); pink[:,:,0] = 255; pink[:,:,2] = 255
    img_with_msk = img * (1 - msk) + alpha * blue * msk +  (1 - alpha) * img * msk
    img_with_lbl = img * (1 - lbl) + alpha * pink * lbl +  (1 - alpha) * img * lbl

    # 1: clean image (filled by inpainting model); 2: image with the artifacts label; 3: image with the hole mask
    imsave(os.path.join(vis_dir, fname), np.hstack([img, img_with_lbl, img_with_msk]))