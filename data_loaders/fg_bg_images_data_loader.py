from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image as PILImage
import torchvision.transforms as transforms
from time import time

def get_file_index_list(path):
    return  [f[f.index("fg_bg_") + 6: f.index(".jpg")] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

class TrainImageDataset(Dataset): 
    

    def __init__(self, root_dir, no_of_batches, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.no_of_batches = no_of_batches
        self.fg_bg_file_index_list = []
        for i in range(no_of_batches):
            self.fg_bg_file_index_list.append(get_file_index_list(self.root_dir + "/batch" + str(i+1) + "/fg_bg_jpg"))

    def __len__(self):
        return 40000 * self.no_of_batches

    def __getitem__(self, idx):
      
        batch_id = int(idx/40000)
        file_id = idx - batch_id*40000
        fg_bg_file_index = self.fg_bg_file_index_list[batch_id][file_id]
        fg_bg_img = PILImage.open(self.root_dir + "/batch" + str(batch_id+1) + "/fg_bg_jpg/fg_bg_" + fg_bg_file_index + ".jpg")
        first_level = fg_bg_file_index[fg_bg_file_index.index("_")+1:]
        bg_index = first_level[:first_level.index("_")]        
        bg_img = PILImage.open(self.root_dir + "/batch" + str(batch_id+1) + "/bg_jpg/bg_img_" + bg_index + ".jpg")
        mask_black_img = PILImage.open(self.root_dir + "/batch" + str(batch_id+1) + "/mask_black_jpg/bg_mask_" + fg_bg_file_index + ".jpg")
        depth_fg_bg_img = PILImage.open(self.root_dir + "/batch" + str(batch_id+1) + "/depth_fg_bg_jpg/depth_fg_bg_" + fg_bg_file_index + ".jpg")
        
        if self.transform is not None:
            bg_img = self.transform(bg_img)
            fg_bg_img = self.transform(fg_bg_img)
            mask_black_img = self.transform(mask_black_img)
            depth_fg_bg_img = self.transform(depth_fg_bg_img)
        
        
        return {"bg" : bg_img, "fg_bg":fg_bg_img, "mask_black": mask_black_img, "depth_fg_bg": depth_fg_bg_img}


class TestImageDataset(Dataset): 
    

    def __init__(self, root_dir, no_of_batches=1, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.no_of_batches = no_of_batches
        self.fg_bg_file_index_list = []
        for i in range(1):
            self.fg_bg_file_index_list.append(get_file_index_list(self.root_dir + "/batch10/fg_bg_jpg"))

    def __len__(self):
        return 40000 

    def __getitem__(self, idx):
        batch_id = int(idx/40000)
        #print("batch_id=" + str(batch_id))
        file_id = idx - batch_id*40000
        #print("file_id=" + str(file_id))
        fg_bg_file_index = self.fg_bg_file_index_list[batch_id][file_id]
        fg_bg_img = PILImage.open(self.root_dir + "/batch10/fg_bg_jpg/fg_bg_" + fg_bg_file_index + ".jpg")
        first_level = fg_bg_file_index[fg_bg_file_index.index("_")+1:]
        bg_index = first_level[:first_level.index("_")]        
        bg_img = PILImage.open(self.root_dir + "/batch10/bg_jpg/bg_img_" + bg_index + ".jpg")
        mask_black_img = PILImage.open(self.root_dir + "/batch10/mask_black_jpg/bg_mask_" + fg_bg_file_index + ".jpg")
        depth_fg_bg_img = PILImage.open(self.root_dir + "/batch10/depth_fg_bg_jpg/depth_fg_bg_" + fg_bg_file_index + ".jpg")
        
        if self.transform is not None:
            bg_img = self.transform(bg_img)
            fg_bg_img = self.transform(fg_bg_img)
            mask_black_img = self.transform(mask_black_img)
            depth_fg_bg_img = self.transform(depth_fg_bg_img)
        
        
        return {"bg" : bg_img, "fg_bg":fg_bg_img, "mask_black": mask_black_img, "depth_fg_bg": depth_fg_bg_img}


