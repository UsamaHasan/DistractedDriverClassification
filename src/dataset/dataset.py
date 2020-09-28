import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from skimage import io
class CustomDataset(Dataset):
    """
    Custom Dataset for Distracted Driver Dataset created by Hesham M.Eraqi
    https://arxiv.org/pdf/1901.09097.pdf#cite.Das2015
    """
    def __init__(self,folder_path,csv_file,mode='train',transform = None):
        """
        Args:
            folder_path(str) : Path to root folder which contains the dataset.
            csv_file(str) : path to csv file which contains the images path and dataset spilt examples.
            mode(str) : default set on train else test format 
            Image                                   Label
            /distracted.driver/Drive Safe/1887.jpg , 0
            transform(torchvision.transform) : default None, else list of transformation to be \
                to each image.
        """
        self.folder_path = folder_path
        self.mode = mode
        self.dataset_file = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataset_file)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        # For clarification check the format of dataset csv defined above.
        img_folder = self.dataset_file.iloc[idx ,0].split('/')[2]
        img_name = self.dataset_file.iloc[idx ,0].split('/')[3]
        img_path = os.path.join(img_folder,img_name)
        img_path = os.path.join(self.folder_path,img_path)
        img = io.imread(img_path)
        label = self.dataset_file.iloc[idx,1]        
        if self.transform:
            img = self.transform(img)
        return img,label
