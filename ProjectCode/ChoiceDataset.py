#This focuses on Cursive and is forbidden from accessing digits.

import os
from PIL import Image
from torch.utils.data import Dataset

class CHoiCeDataset(Dataset):
    def __init__(self, root_dir, transform=None, exclude_digits=False):
        self.root_dir = root_dir
        self.transform = transform
        self.exclude_digits = exclude_digits
        self.samples = []


        data_dir = os.path.join(root_dir, 'data')

        if os.path.exists(data_dir):
            # CHoiCe format: numeric folders 0-61
            for label in range(62):
                # Skip digits (labels 0-9) if exclude_digits is True
                if self.exclude_digits and label < 10:
                    continue

                label_dir = os.path.join(data_dir, str(label))
                if os.path.exists(label_dir):
                    for fname in os.listdir(label_dir):
                        if fname.lower().endswith('.png'):
                            self.samples.append(
                                (os.path.join(label_dir, fname), label)
                            )
        else:
            # Fallback: original character-based folders
            for root, dirs, files in os.walk(root_dir):
                dirname = os.path.basename(root)
                if len(dirname) != 1:
                    continue

                label_char = dirname

                # EMNIST mapping:
                if label_char.isdigit():
                    label = ord(label_char) - ord('0')  # 0–9
                elif label_char.isupper():
                    label = ord(label_char) - ord('A') + 10  # 10–35
                elif label_char.islower():
                    label = ord(label_char) - ord('a') + 36  # 36–61
                else:
                    continue

                for fname in files:
                    if fname.lower().endswith('.png'):
                        self.samples.append(
                            (os.path.join(root, fname), label)
                        )

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label
