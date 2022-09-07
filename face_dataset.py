import os

import PIL.Image
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection

class FaceDataset(Dataset):
    def __init__(self, is_train=True):
        self.GENDER_BOUNDARY = 7380
        self.img_dir = "data/images"
        self.is_train = is_train

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((300,300)),
            transforms.Resize(64)
        ])
        faces = os.listdir(self.img_dir)
        self.image_list = []
        self.age_list = []
        self.gender_list = []
        i = 0
        for face in faces:
            serial_str = face[:5]
            serial = int(serial_str)
            gender = 0
            if serial > self.GENDER_BOUNDARY:
                gender = 1
            age_str = face[6:8]
            age = float(age_str)
            self.image_list.append(face)
            self.age_list.append(age)
            self.gender_list.append(gender)
            i = i + 1

        self.indices = list(range(len(self.image_list)))
        train, test = model_selection.train_test_split(self.indices, test_size=0.2)
        self.data = train
        if not self.is_train:
            self.data = test
        self.__scale__()

    def __scale__(self):
        labels = [[float(i)] for i in self.age_list]
        self.scaler = MinMaxScaler()
        labels = self.scaler.fit_transform(labels)
        labels = torch.tensor(labels, dtype=torch.float32)
        self.age_torch_list = torch.squeeze(labels)

    def unscale(self, values):
        values = [[i] for i in values]
        values = self.scaler.inverse_transform(values)
        values = [i[0] for i in values]
        return values

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        age_torch = self.age_torch_list[idx]
        gender = self.gender_list[idx]
        gender_torch = torch.tensor(gender)
        img_path = os.path.join(self.img_dir, image_name)
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        return image, age_torch, gender_torch

if __name__ == "__main__":
    cid = FaceDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)

    print(cid.age_list)
    print(cid.age_torch_list)
    print("unscaled")
    print(cid.unscale([100]))
    print(cid.unscale([1,2,100,200]))

    for image, age, gender in dataloader:
        print(image.shape)
        print(age)
        print(gender)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)

