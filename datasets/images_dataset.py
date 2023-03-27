import torch
import sparse
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
        if opts.dataset_type == 'CosMx':
            self.ext = 'flo.png'
        elif opts.dataset_type == 'Xenium':
            self.ext = 'hne.png'
        elif opts.dataset_type == 'Visium':
            self.ext = '.npz'

        self.paths = list(Path(target_root).rglob(f'*{self.ext}'))
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.opts.dataset_type in ('CosMx', 'Xenium'):
            img = Image.open(str(self.paths[index]))
            img = img.resize((64, 64), 
                             resample=Image.Resampling.BICUBIC)
            img = img.resize((128, 128), 
                             resample=Image.Resampling.BICUBIC)
            img = np.array(img).clip(0, 255).astype(np.uint8)
            img = self.target_transform(img)
            rna = str(self.paths[index]).replace(self.ext, 'rna.npz')
            rna = sparse.load_npz(rna).sum((0, 1)).todense()
            rna = torch.from_numpy(rna).to(img).float()
            if self.opts.dataset_type == 'Xenium':
                rna = rna[:self.opts.rna_num]
        elif self.opts.dataset_type == 'Visium':
            npz = np.load(str(self.paths[index]))
            img = npz['img'][96:-96, 96:-96]
            img = Image.fromarray(img)
            # the resize step is inspired by clean-FID
            img = img.resize((128, 128), resample=Image.Resampling.BICUBIC)
            img = np.asarray(img).clip(0, 255).astype(np.uint8)
            img = self.target_transform(img)
            rna = npz['key_melanoma_marker']
            rna = torch.from_numpy(rna).to(img).float()

        # here the first im is a legacy input
        # which is not very useful in our case
        return img, img, rna
