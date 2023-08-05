import pandas as pd
from PIL import Image
from torch.utils import data
import os
from torchvision import transforms
import matplotlib.image as img
import matplotlib.pyplot as plt

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "test18":
        df = pd.read_csv('test18.csv')
        path = df['Path'].tolist()
        return path
    elif mode == "test50":
        df = pd.read_csv('test50.csv')
        path = df['Path'].tolist()
        return path
    elif mode == "test152":
        df = pd.read_csv('test152.csv')
        path = df['Path'].tolist()
        return path

class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.mode = mode
        if (self.mode == "train") or (self.mode == "valid"):
            self.img_name, self.label = getData(mode)
        else:
            self.img_name = getData(mode)

        self.img_name = [name[1:] for name in self.img_name]
        
        print("> Found %d images..." % (len(self.img_name)))  
        transform_list = []
        transform_list.append(transforms.RandomRotation(degrees=30))
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.CenterCrop(300))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]))
        self.transform = transforms.Compose(transform_list)

        self.current_path = os.path.abspath(os.getcwd())

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""
        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        
        path = self.current_path + self.img_name[index]
        img = Image.open(path).convert('RGB')
        '''
        plt.imshow(img)
        plt.title("Original Image")
        plt.show()
        '''
       #transform image
        
        
        img = self.transform(img)

        if (self.mode == "train") or (self.mode == "valid"):
            label = self.label[index]
            return img, label, path
        else:
            return img, self.img_name[index]
        
        