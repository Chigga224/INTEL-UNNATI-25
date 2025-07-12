from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir='data/', transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img