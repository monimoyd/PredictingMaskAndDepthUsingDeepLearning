import torch
import torchvision
import numpy as np
import skimage
from skimage.transform import resize
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

def draw_and_save(tensors, output_full_path, figsize = [20,20], display=True, display_with_plasma=False, *args, **kargs):
    try:
        tensors = tensors.detach().cpu()
    except:
        pass

    imgs_to_display = tensors[0:8,:,:]
   
    plt.figure(figsize=figsize)
    if display_with_plasma==True:
        plasma = plt.get_cmap('plasma') 
        imgs = []
        for i in range(imgs_to_display.shape[0]):
            display_image = imgs_to_display[i]          
            rescaled = (display_image[0, :,:]).numpy()
            imgs.append(plasma(rescaled)[:,:,:3])
        img_set = np.stack(imgs)
        mon =  skimage.util.montage(img_set, multichannel=True, fill=(1,1,1), grid_shape=(1,8))
        plt.imshow((mon * 255).astype(np.uint8))

    else:
        grid_tensors = make_grid(imgs_to_display, 8)
        grid_images = grid_tensors.permute(1,2,0)
        plt.imshow(torch.clamp(grid_images, 0.0, 1.0))

    plt.axis("off")
    fig = plt.gcf()
    if display==True:
        plt.show()
    fig.savefig(output_full_path)
