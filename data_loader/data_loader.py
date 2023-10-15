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
    def __init__(self, csv_file, root_dir, scale=2, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        self.scale = scale
        print("[+] Data Loaded with rows: ", super().__len__())

    ### Get item returns an image for specific idx
    def __getitem__(self, idx):
        lr_image_name = self.csv_dataframe.iloc[idx, 0]
        hr_image_name = self.csv_dataframe.iloc[idx, 1]
        lr_image = cv2.imread(lr_image_name)
        hr_image = cv2.imread(hr_image_name)
        lr_image = lr_image.astype(dtype=np.float32) / 255
        hr_image = hr_image.astype(dtype=np.float32) / 255
        sample = {'lr_image':lr_image, 'hr_image': hr_image}
        if self.transform:
            sample = self.transform(sample)
        return sample