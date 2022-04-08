from torch import from_numpy
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from utils import data_utils

import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesDataset(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        to_path = self.target_paths[index]
        from_im = Image.open(from_path)
        to_im = Image.open(to_path)
        if 'ham10k' in str(from_path):
            from_im = from_im.resize((64, 64)).resize((256, 256))
            to_im = to_im.resize((64, 64)).resize((256, 256))

        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im