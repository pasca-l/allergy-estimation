import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image


class FoodDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='../data/'):
        super().__init__()
        self.data_dir = data_dir
        # transformer to fit requirement of pre-trained network
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        full_dataset =\
            FoodDataset(self.data_dir, phase='train', trans=self.transform)
        self.train_data, self.val_data =\
            random_split(full_dataset, [55000, 5000])

        # self.test_data =\
        #     FoodDataset(self.data_dir, phase='test', trans=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batchsize=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batchsize=8, shuffle=True)


class FoodDataset(Dataset):
    def __init__(self, data_dir, phase, trans):
        self.phase = phase
        self.transform = trans
        self.image_paths = [] # from data_dir?

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        tranformed_img = self.transform(img)

        label = 0 # label from data

        return transformed_img, label
