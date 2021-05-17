import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def denormalize(image):
    image = (image + 1) / 2.0 * 255.
    image = torch.round(image) / 255.
    image = torch.clamp(image, 0., 1.)
    return image

def visualize_single_map(mapi, name):
    if not os.path.exists('visualizations/'):
            os.makedirs('visualizations/')
    x = torch.stack([mapi[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    x = x.permute(1,2,0)
    x = np.uint8(x)
    plt.imsave('visualizations/{}.jpg'.format(name), np.squeeze(x))

def save_image(img, name, args):
    if not os.path.exists('visualizations/'):
            os.makedirs('visualizations/')
    x = denormalize(img[0].cpu(), args) 
    x = x.permute(1, 2, 0).numpy()
    plt.imsave('visualizations/'+name+'.jpg', x)

def visualize_run(i, img, mask, erased_img, pred_img, target_img):
    plt.rcParams.update({'font.size': 10})

    img = denormalize(img[0].cpu()) 
    img = img.permute(1, 2, 0).numpy()

    mask = torch.stack([mask[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    mask = mask.permute(1, 2, 0).numpy()
    mask = np.uint8(mask)

    erased_img = denormalize(erased_img[0].cpu()) 
    erased_img = erased_img.permute(1, 2, 0).numpy()

    pred_img = denormalize(pred_img[0].cpu()) 
    pred_img = pred_img.permute(1, 2, 0).numpy()

    target_img = denormalize(target_img[0].cpu()) 
    target_img = target_img.permute(1, 2, 0).numpy()

    f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize=(12, 3))
    
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(mask)
    ax2.set_title("Mask")
    ax2.axis('off')
    
    ax3.imshow(erased_img)
    ax3.set_title("Incomplete Image")
    ax3.axis('off')

    ax4.imshow(pred_img)
    ax4.set_title("Predicted Image")
    ax4.axis('off')

    ax5.imshow(target_img)
    ax5.set_title("Inpainted Image")
    ax5.axis('off')

    f.savefig('visualizations/runs/run' + str(i) + '.jpg')
    plt.close(f)

def visualize_test(i, img, erased_img, pred_img, target_img):
    plt.rcParams.update({'font.size': 10})

    img = denormalize(img[0].cpu()) 
    img = img.permute(1, 2, 0).numpy()

    erased_img = denormalize(erased_img[0].cpu()) 
    erased_img = erased_img.permute(1, 2, 0).numpy()

    pred_img = denormalize(pred_img[0].cpu()) 
    pred_img = pred_img.permute(1, 2, 0).numpy()

    target_img = denormalize(target_img[0].cpu()) 
    target_img = target_img.permute(1, 2, 0).numpy()

    f, (ax1,ax3,ax4,ax5) = plt.subplots(1, 4, figsize=(9, 3))
    
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax3.imshow(erased_img)
    ax3.set_title("Incomplete Image")
    ax3.axis('off')

    ax4.imshow(pred_img)
    ax4.set_title("Predicted Image")
    ax4.axis('off')

    ax5.imshow(target_img)
    ax5.set_title("Inpainted Image")
    ax5.axis('off')

    f.savefig('visualizations/demos/demo' + str(i) + '.jpg')
    plt.close(f)