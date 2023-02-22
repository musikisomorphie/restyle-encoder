import numpy as np

from pathlib import Path
from utils import data_utils
from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesDataset_RxRx19(Dataset):

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
        from_im = np.array(Image.open(from_path))
        to_im = np.array(Image.open(to_path))

        if 'rxrx19' in self.opts.dataset_type:
            col = from_im.shape[1] // 2
            from_im = np.concatenate((from_im[:, :col],
                                      from_im[:, col:]), axis=-1)
            to_im = np.concatenate((to_im[:, :col],
                                    to_im[:, col:]), axis=-1)
            if 'rxrx19a' == self.opts.dataset_type:
                from_im = from_im[:, :, :-1]
                to_im = to_im[:, :, :-1]

            if self.opts.input_ch != -1:
                from_im = np.expand_dims(from_im[:, :, self.opts.input_ch], -1)
                to_im = np.expand_dims(to_im[:, :, self.opts.input_ch], -1)

        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im


class ImagesDataset(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
        assert source_transform is None
        if 'rxrx19' in opts.dataset_type:
            exts = '*.png'
        elif opts.dataset_type == 'CosMx':
            exts = '*_cell.png'
        elif opts.dataset_type == 'Xenium':
            exts = '*_hne.png'

        self.paths = list(Path(target_root).rglob(exts))
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        im = Image.open(str(self.paths[index]))
        im = np.array(im)

        if 'rxrx19' in self.opts.dataset_type:
            col = im.shape[1] // 2
            im = np.concatenate((im[:, :col],
                                 im[:, col:]), axis=-1)
            if self.opts.dataset_type == 'rxrx19a':
                im = im[:, :, :-1]

            if self.opts.input_ch != -1:
                im = np.expand_dims(im[:, :, self.opts.input_ch], -1)

        if self.target_transform:
            im = self.target_transform(im)

        # here the first im is a legacy input
        # which is not very useful in our case
        return im, im
