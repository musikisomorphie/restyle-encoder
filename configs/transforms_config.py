import torch
from abc import abstractmethod
import torchvision.transforms as transforms


class TransformsConfig(object):

    def __init__(self, opts):
        self.opts = opts

    @abstractmethod
    def get_transforms(self):
        pass


class EncodeTransforms(TransformsConfig):

    def __init__(self, opts):
        super(EncodeTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict


class CarsEncodeTransforms(TransformsConfig):

    def __init__(self, opts):
        super(CarsEncodeTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict


class MedTransforms(TransformsConfig):

    def __init__(self, opts):
        super(MedTransforms, self).__init__(opts)
        img_chn = 3
        if 'rxrx19' in self.opts.dataset_type:
            if self.opts.input_ch == -1:
                img_chn = 5 if self.opts.dataset_type == 'rxrx19a' else 6
            else:
                img_chn = 1
        # img_chn = 6 if 'rxrx19b' in self.opts.dataset_type else 3
        self.mean = [0.5] * img_chn + [0] * opts.rna_num
        self.std = [0.5] * img_chn + [1] * opts.rna_num

    def get_transforms(self):
        def to_tensor(x):
            if isinstance(x, list):
                # img divided by 255 while rna remains unchanged
                x[0] = transforms.functional.to_tensor(x[0])
                x[1] = transforms.functional.to_tensor(x[1])
                x = torch.cat([x[0], x[1]], 0)
            else:
                x = transforms.functional.to_tensor(x)
            return x
        t_tensor = transforms.Lambda(lambda x: to_tensor(x))

        angles = [0, 90, 180, 270]

        def random_rotation(x):
            angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
            if angle > 0:
                x = transforms.functional.rotate(x, angle)
            return x
        t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                t_tensor,
                t_random_rotation,
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(self.mean, self.std, inplace=True)]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                t_tensor,
                transforms.Normalize(self.mean, self.std, inplace=True)]),
            'transform_inference': transforms.Compose([
                t_tensor,
                transforms.Normalize(self.mean, self.std, inplace=True)]),
        }
        return transforms_dict
