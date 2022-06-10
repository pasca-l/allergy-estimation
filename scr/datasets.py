import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from PIL import Image


class FoodDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='../data/images/', ann_dir='../data/meta/'):
        super().__init__()
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.ToPILImage(),
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset =\
                FoodDataset(self.data_dir, self.ann_dir, 'train', self.trans)
            train_size = int(len(full_dataset) * 0.8)
            val_size = len(full_dataset) - train_size

            self.train_data, self.val_data =\
                random_split(full_dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test_data =\
                FoodDataset(self.data_dir, self.ann_dir, 'test', self.trans)

        if stage == "predict":
            self.predict_data = None

    def train_dataloader(self):
        return DataLoader(self.train_data, batchsize=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batchsize=8, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batchsize=8)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batchsize=8)


class FoodDataset(Dataset):
    def __init__(self, data_dir, ann_dir, phase, trans):
        self.data_dir = data_dir
        self.transform = trans

        ann_file = os.path.join(ann_dir, f"{phase}.txt")
        self.img_list = open(ann_file, 'r').read().splitlines()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.data_dir, f"{img_name}.jpg")
        pil_img = Image.open(img_path).convert("RGB")
        transformed_img = self.transform(pil_img)

        label = img_name.split('/')[0]

        return np.array(transformed_img), label
