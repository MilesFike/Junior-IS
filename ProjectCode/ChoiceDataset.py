import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CHoiCeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for fname in os.listdir(root_dir):
            if fname.endswith('.png'):
                label_char = fname.split('_')[0]
                if label_char.isdigit():
                    label = ord(label_char) - ord('0') + 52  #0,9 52,61
                elif label_char.isupper():
                    label = ord(label_char) - ord('A')  # A,Z 0,25
                elif label_char.islower():
                    label = ord(label_char) - ord('a') + 26  # a,z 26-51
                else:
                    continue
                self.samples.append((os.path.join(root_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

