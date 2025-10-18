import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.A_dir = os.path.join(root_dir, split, "A")
        self.B_dir = os.path.join(root_dir, split, "B")
        self.filenames = sorted(os.listdir(self.A_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_A = Image.open(os.path.join(self.A_dir, fname)).convert("RGB")
        img_B = Image.open(os.path.join(self.B_dir, fname)).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B