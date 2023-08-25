import os
import numpy as np
import json 
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

def get_label_order():
    root = os.getcwd()
    path = root + "/file/objects.json"
    label_list = []
    with open(path) as file:
        label_dic = json.load(file)
    for label in label_dic:
        label_list.append(label)

    return label_list

def get_training_data():
    root = os.getcwd()
    path = root + "/file/train.json"
    with open(path) as file:
        data = json.load(file)
    img_list = []
    label_list = []
    string_list = get_label_order()

    for img, labels in data.items():
        img_list.append(img)
        temp = [0]*24
        for label in labels:     
            temp[string_list.index(label)] += 1
        label_list.append(temp)
    label_list = torch.tensor(np.array(label_list))
    return img_list, label_list

def get_testing_data(file_name):
    root = os.getcwd()
    path = root + "/file/" + file_name
    with open(path) as file:
        data = json.load(file)
    print(data)
    label_list = []
    string_list = get_label_order()

    for labels in data:
        print(labels)
        temp = [0]*24
        for label in labels:     
            temp[string_list.index(label)] += 1
        label_list.append(temp)
    label_list = torch.tensor(np.array(label_list))
    return label_list

class get_picture():
    def __init__(self, mode):
        self.root = os.getcwd() + "/iclevr/"
        self.mode = mode
        if mode=="train":
            self.img_name, self.label = get_training_data()
        elif mode=="test" or mode=="new_test":
            self.label = get_testing_data(mode+".json")

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            path = self.root +self.img_name[index]
            transform=transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
            im = Image.open(path) 
            im.show() 
            print(j)
            img = transform(Image.open(path).convert('RGB'))
        else:
            img = torch.ones(1)

        label = self.label[index]
        return img, label

def save_images(images, name):
    save_image(images, fp = "./"+name+".png")

def main():
    test = get_picture("train")
    test.__getitem__(0)

    
if __name__ == '__main__':
    main()