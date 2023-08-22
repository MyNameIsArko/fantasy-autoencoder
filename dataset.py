from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class FantasySet(Dataset):
    def __init__(self, data_path='./data'):
        self.img_paths = [os.path.join(data_path, img_name) for img_name in os.listdir(data_path)]
        self.transforms = Compose([Resize(256), ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = self.transforms(img)
        return img
