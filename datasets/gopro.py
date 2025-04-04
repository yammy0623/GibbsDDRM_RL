import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class GoProDataset(Dataset):
    def __init__(self, root_dir, data_type, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = [] 

        for scene in os.listdir(root_dir):
            blur_folder = os.path.join(root_dir, scene, data_type)
            sharp_folder = os.path.join(root_dir, scene, 'sharp')

            if not os.path.isdir(blur_folder) or not os.path.isdir(sharp_folder):
                continue

            for img_name in sorted(os.listdir(blur_folder)):
                blur_path = os.path.join(blur_folder, img_name)
                sharp_path = os.path.join(sharp_folder, img_name)
                if os.path.exists(sharp_path):
                    self.image_pairs.append((blur_path, sharp_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.image_pairs[idx]

        blur_img = cv2.imread(blur_path, cv2.IMREAD_COLOR)
        sharp_img = cv2.imread(sharp_path, cv2.IMREAD_COLOR)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img

