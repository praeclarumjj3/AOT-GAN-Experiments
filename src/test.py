import os
import argparse
import importlib
import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from utils.helpers import visualize_test
from utils.option import args 
from tqdm import tqdm
import torchvision.transforms as transforms
from icecream import ic

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def main_worker(args, use_gpu=True): 

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()

    # prepare dataset
    image_paths = []
    with open(os.path.join(args.dir_image, args.data_test,'val.txt')) as f:
        images_list = f.read().splitlines()
    for path in images_list: 
        image_paths.append(os.path.join(args.dir_image, args.data_test,path))
    # image_paths.sort()
    mask_paths = glob(os.path.join(args.dir_mask, args.mask_type, '*.png'))
    os.makedirs(args.outputs, exist_ok=True)
    
    trans = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),])
    j = 0

    # iteration through datasets
    for ipath, mpath in tqdm(zip(image_paths, mask_paths)): 
        
        if j >= args.num_test:
            exit()

        image = ToTensor()(Image.open(ipath).convert('RGB'))
        image = trans(image)
        image = (image * 2.0 - 1.0).unsqueeze(0)
        mask = ToTensor()(Image.open(mpath).convert('L'))
        mask = trans(mask)
        mask = mask.unsqueeze(0)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask.float()) + mask
        
        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img
    
        visualize_test(j, image, image * (1-mask), pred_img.detach(), comp_imgs.detach())
        j+=1


if __name__ == '__main__':
    main_worker(args)
