import torch
from torch.utils.data import Dataset
from PIL import Image
import sys
import NoiseManager
from NoiseManager import Noises
import numpy as np
from skimage import exposure
import random
class NoisyDataset(Dataset):
    def __init__(self, image_paths, transform=None,Test=False):
        self.image_paths = image_paths
        self.transform = transform
        self.Test=Test
    
    
    def __len__(self):
        return len(self.image_paths)
    
    def collate_fn(batch):
        """
        this function is used in the dataloader objecct to customize the way of working wiht batches.

        :param batch: an iterable of N sets from __getitem__()
        :return: three tensors, one for the images, one for the enhanced images and the last one for noisy images.
        """
        images = list()
        
        targets=list()
        for b in batch:
            images.append(b[0])
            targets.append(torch.b[1])
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)
        return images,targets


    def get_item(self,index):
        return self.__getitem__(index)


    def __getitem__(self, index):
        """this function will return three imges. The original image, an enhanced image using histogram equalization, and a noisy image
        where the noise can be one of several types including Gaussian nois, salt and pepper nois, speckle nois ..."""
        image_path = self.image_paths[index]
        image = np.array(Image.open(image_path).convert('L'))
        img_class=str(image_path.split("/")[-3]).lower()
        if img_class == "Speckle_Noise".lower():
            target=(1,0,0)
        elif img_class == "Salt_Pepper".lower():
            target=(0,1,0)
        elif img_class == "Uneven_Illumination".lower():
            target=(0,0,1)
        else:
            print("error")
            sys.exit(0)
        #image = exposure.equalize_adapthist(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if not self.Test:
           
            enhanced_image=NoiseManager.make_hsv_equalized(image,hsv=False)
            rand=np.random.rand()
            
            if rand<0.25:
                noisy_image=NoiseManager.noisy(Noises.NOIS_GAUSS,enhanced_image,var=rand,sigma=0.5)
                NoiseType=Noises.NOIS_GAUSS
            elif rand<0.5:
                noisy_image=NoiseManager.noisy(Noises.NOIS_SALT_AND_PEPPER,enhanced_image,s_vs_p=0.5,amount=rand/2)
                NoiseType=Noises.NOIS_SALT_AND_PEPPER
            elif rand<0.75:
                noisy_image=NoiseManager.noisy(Noises.NOIS_POISSON,enhanced_image)
                NoiseType=Noises.NOIS_POISSON
            else:
                noisy_image=NoiseManager.noisy(Noises.NOIS_SPECKLE,enhanced_image)
                NoiseType=Noises.NOIS_POISSON
            
            return image,noisy_image,NoiseType
        else:
            return image,target