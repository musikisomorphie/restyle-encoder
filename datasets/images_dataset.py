import torch
import random
import sparse
import numpy as np

from pathlib import Path
from utils import data_utils
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from argparse import Namespace

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


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
            rna = str(self.paths[index]).replace(self.ext, 'rna.npz')
            rna = sparse.load_npz(rna)
            if self.opts.dataset_type == 'Xenium':
                rna = rna[:, :, :self.opts.rna_num]
            out = self.target_transform([img, rna.todense()])
            img, rna = out[:3], out[3:]
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


if __name__ == '__main__':
    opts = Namespace()
    # opts.dataset_type = 'CosMx'
    # opts.rna_num = 1000

    opts.dataset_type = 'Xenium'
    opts.rna_num = 280

    def to_tensor(x):
        if isinstance(x, list):
            # img divided by 255 while rna remains unchanged
            x[0] = F.to_tensor(x[0])
            x[1] = F.to_tensor(x[1])
            x = torch.cat([x[0], x[1]], 0)
        else:
            x = F.to_tensor(x)
        return x
    t_tensor = transforms.Lambda(lambda x: to_tensor(x))

    angles = [0, 90, 180, 270]

    def random_rotation(x):
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = transforms.functional.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    mean = [0.5 for _ in range(3)] + [0 for _ in range(opts.rna_num)]
    std = [0.5 for _ in range(3)] + [1 for _ in range(opts.rna_num)]

    # def norm(x, split=3):
    #     x = F.normalize(x, mean, std, inplace=True)
    #     if split is not None:
    #         return x[:3], x[3:]
    #     else:
    #         return x

    # t_norm = transforms.Lambda(lambda x: norm(x))

    # def random_rotation(x):
    #     t = random.randint(0, 3)
    #     if t > 0:
    #         if isinstance(x, list):
    #             x[0] = F.rotate(x[0], t * 90)
    #             for _ in range(t):
    #                 # reverse the column
    #                 x[1].coords[1] = x[1].shape[1] - 1 - x[1].coords[1]
    #                 # swap the x, y pos
    #                 x[1].coords[1], x[1].coords[0] = x[1].coords[0], x[1].coords[1]
    #         else:
    #             x = F.rotate(x, t * 90)
    #     return x
    # t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    # def h_flip(x, p=0.5):
    #     if torch.rand(1) < p:
    #         if isinstance(x, list):
    #             x[0] = F.hflip(x[0])
    #             x[1].coords[1] = x[1].shape[1] - 1 - x[1].coords[1]
    #             return x
    #         else:
    #             return F.hflip(x)
    #     else:
    #         return x
    # t_hflip = transforms.Lambda(lambda x: h_flip(x))
    
    trans = transforms.Compose([
        t_tensor,
        t_random_rotation,
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std, inplace=True)
    ])

    pth = Path('Data') / opts.dataset_type / 'GAN' / 'crop'
    idt = ImagesDataset(source_root=None, target_root=pth,
                        opts=opts, target_transform=trans)
    dload = DataLoader(idt,
                       batch_size=8,
                       shuffle=False,
                       num_workers=8,
                       drop_last=True)

    for did, (img, rn, rna) in enumerate(dload):
        # this does not work after freezing the dataset class
        rn = rn.sum((-1, -2))
        rna = rna.sum((1, 2))
        print(did, img.shape, (rn == rna).all())
        if did == 100:
            break
