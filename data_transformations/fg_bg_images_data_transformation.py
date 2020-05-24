
import torchvision.transforms as transforms
import numpy as np
import torch



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., prob=0.2):
        self.std = std
        self.mean = mean
        self.prob = prob
        
    def __call__(self, tensor):
        rn = np.random.random()
        if rn < self.prob: 
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_train_transform():
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.075, contrast=0.075, saturation=0.075, hue=0.075),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.05, 0.2)
    ])
    return transform

def get_test_transform():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform
