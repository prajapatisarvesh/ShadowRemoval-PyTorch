'''
LAST UPDATE: 2023.09.20
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 


'''
from base.base_dataloader import BaseDataLoader
from torchvision import datasets, transforms
from skimage import io
import cv2
import numpy as np
### Data loader class that parses the CSV, abstraction of BaseDataLoader
class ISTDLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        print("[+] Data Loaded with rows: ", super().__len__())

    ### Get item returns an image for specific idx
    def __getitem__(self, idx):
        shadow = self.csv_dataframe.iloc[idx, 0]
        shadow_mask = self.csv_dataframe.iloc[idx, 1]
        shadow_free = self.csv_dataframe.iloc[idx, 2]
        shadow_image = cv2.imread(shadow)
        shadow_mask_image = cv2.imread(shadow_mask)
        shadow_free_image = cv2.imread(shadow_free)
        shadow_image = shadow_image.astype(dtype=np.float32) / 255
        shadow_mask_image = shadow_mask_image.astype(dtype=np.float32) / 255
        shadow_free_image = shadow_free_image.astype(dtype=np.float32) / 255
        sample = {'shadow_image':shadow_image, 
                  'shadow_mask_image': shadow_mask_image,
                  'shadow_free_image': shadow_free_image,
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample