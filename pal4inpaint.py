# AdobePatentID="11288

import torch
import torch.optim as optim
import torch.nn.functional as F
import pdb
from PIL import Image
import numpy as np 
import cv2
from skimage.io import imsave
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import glob
import os

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]

def get_mean_stdinv(img):
    mean_img = np.zeros((img.shape))
    mean_img[:,:,0] = mean[0]
    mean_img[:,:,1] = mean[1]
    mean_img[:,:,2] = mean[2]
    mean_img = np.float32(mean_img)
    std_img = np.zeros((img.shape))
    std_img[:,:,0] = std[0]
    std_img[:,:,1] = std[1]
    std_img[:,:,2] = std[2]
    std_img = np.float64(std_img)
    stdinv_img = 1 / np.float32(std_img)
    return mean_img, stdinv_img

def numpy2tensor(img):
    img = torch.from_numpy(img).transpose(0,2).transpose(1,2).unsqueeze(0).float()
    return img

def prepare_img(img_file, device):
    img_np = np.array(Image.open(img_file)); H, W = img_np.shape[0], img_np.shape[1]
    mean_img, stdinv_img = get_mean_stdinv(img_np)
    img_tensor = numpy2tensor(img_np).to(device)
    mean_img_tensor = numpy2tensor(mean_img).to(device)
    stdinv_img_tensor = numpy2tensor(stdinv_img).to(device)
    img_tensor = img_tensor - mean_img_tensor
    img_tensor = img_tensor * stdinv_img_tensor
    return img_np, img_tensor

def inference_on_image(model, img_tensor):
    seg_logit = model(img_tensor)
    seg_pred = seg_logit.argmax(dim = 1)
    seg_pred_np = seg_pred.cpu().data.numpy()[0]
    return seg_pred_np

def overlay_seg_on_img(img_np, seg_pred_np, alpha = 0.3):
    pink = np.zeros((img_np.shape)); pink[:,:,0] = 255; pink[:,:,2] = 255
    if len(seg_pred_np.shape) == 2:
        seg_pred_np = np.repeat(np.expand_dims(seg_pred_np, 2), 3, 2)
    vis = img_np * (1 - seg_pred_np) + alpha * pink * seg_pred_np + (1 - alpha) * img_np * seg_pred_np
    return vis



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ckpt_file', default='./ckpt/best_mIoU_iter_1800_torchscript.pth', type=str)

    # arguments needed for single image testing
    parser.add_argument('--img_file', default='', type=str)
    parser.add_argument('--output_seg_file', default='', type=str)
    parser.add_argument('--output_vis_file', default='', type=str)

    # arguments needed for batch testing
    parser.add_argument('--img_dir', default='', type=str)
    parser.add_argument('--output_seg_dir', default='', type=str)
    parser.add_argument('--output_vis_dir', default='', type=str)

    args = parser.parse_args()
    
    # Load the Perceptual Artifacts Localization network
    model = torch.load(args.ckpt_file).to(args.device)

    # Process images
    if args.img_file is not '':

        print('Process a single image...')

        os.makedirs(os.path.dirname(args.output_seg_file), exist_ok = True)
        os.makedirs(os.path.dirname(args.output_vis_file), exist_ok = True)
        
        fname = os.path.basename(args.img_file).split('.')[0]
        img_np, img_tensor = prepare_img(args.img_file, args.device)
        seg_pred_np = inference_on_image(model, img_tensor)
        vis_np = overlay_seg_on_img(img_np, seg_pred_np)

        imsave(args.output_seg_file, seg_pred_np)
        imsave(args.output_vis_file, np.hstack([img_np, vis_np]))

    elif args.img_dir is not '': 

        print('Process a batch of images...')

        os.makedirs(args.output_seg_dir, exist_ok = True)
        os.makedirs(args.output_vis_dir, exist_ok = True)

        for img_file in tqdm(glob.glob(args.img_dir + '/*')):
            fname = os.path.basename(img_file)
            img_np, img_tensor = prepare_img(img_file, args.device)
            seg_pred_np = inference_on_image(model, img_tensor)
            vis_np = overlay_seg_on_img(img_np, seg_pred_np)

            imsave(os.path.join(args.output_seg_dir, fname), seg_pred_np)
            imsave(os.path.join(args.output_vis_dir, fname), np.hstack([img_np, vis_np]))
        
    else:
        raise NotImplementedError
