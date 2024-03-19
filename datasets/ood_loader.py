import os
import torch
import torchvision

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST, KMNIST

DATA_PATH = './'
OOD_DATA_PATH = DATA_PATH
IMAGENET_PATH = DATA_PATH
TINYIMAGENET_PATH = DATA_PATH


class CIFAR10(object):
    def __init__(self, data_root=DATA_PATH, batch_size=128, img_size=32, options=None):
        print(options)
        train_transform = pre_trans(options['pre_trans'], options['manual_ctr'], img_size=img_size)
        transform = test_transform(img_size=img_size)

        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'])

        test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        options['img_size'] = 32
        options['num_known'] = 10


class CIFAR100(object):
    def __init__(self, data_root=DATA_PATH, batch_size=128, img_size=32, options=None):
        
        train_transform = pre_trans(options['pre_trans'], options['manual_ctr'], img_size=img_size)
        transform = test_transform(img_size=img_size)

        train_set = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'])

        test_set = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])

        self.train_loader = train_loader
        self.test_loader  = test_loader
        
        options['img_size'] = 32
        # options['num_known'] = 100
        

class SVHN(object):
    def __init__(self, data_root=DATA_PATH, batch_size=128, img_size=32, options=None):

        train_transform = pre_trans(options['pre_trans'], options['manual_ctr'], img_size=img_size)
        transform = test_transform(img_size=img_size)

        train_set = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'])

        test_set = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])

        self.train_loader = train_loader
        self.test_loader  = test_loader


class OOD_Dataset(Dataset):
    def __init__(self, folder_path, img_size=32):
        self.transforms = test_transform(img_size)
        file_list = os.listdir(folder_path)
        self.img_paths = []
        for file in file_list:
            if(file.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']):
                self.img_paths.append(os.path.join(folder_path, file))

    def __len__(self):
        print(len(self.img_paths))
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        label = -1
        img = self.transforms(img)
        return img, label


class LSUN(object):
    def __init__(self, options):
        batch_size = options['batch_size']
        data_root  = os.path.join(OOD_DATA_PATH, 'LSUN')
        
        test_set = OOD_Dataset(folder_path=data_root)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])

        self.test_loader = test_loader


class LSUN_R(object):
    def __init__(self, options):
        batch_size = options['batch_size']
        data_root  = os.path.join(OOD_DATA_PATH, 'LSUN_resize')
        
        test_set = OOD_Dataset(folder_path=data_root)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])

        self.test_loader = test_loader


class IMGN_R(object):
    def __init__(self, options):
        batch_size = options['batch_size']
        data_root  = os.path.join(OOD_DATA_PATH, 'Imagenet_resize')
        
        test_set = OOD_Dataset(folder_path=data_root)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])

        self.test_loader = test_loader


class IMGN(object):
    def __init__(self, options):
        batch_size = options['batch_size']
        data_root  = os.path.join(OOD_DATA_PATH, 'Imagenet')
        
        test_set = OOD_Dataset(folder_path=data_root)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])
        
        self.test_loader = test_loader
        
        
class iNaturalist(object):
    def __init__(self, options):
        batch_size = options['batch_size']
        data_root  = os.path.join(OOD_DATA_PATH, 'iNaturalist')
        
        test_set = OOD_Dataset(folder_path=data_root, img_size=224)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'])
        
        self.test_loader = test_loader


__factory = {
    'svhn'    : SVHN,
    'mnist'   : MNIST,
    'kmnist'  : KMNIST,
    'cifar10' : CIFAR10,
    'cifar100': CIFAR100,

    'LSUN'  : LSUN,
    'LSUN_R': LSUN_R,
    'IMGN'  : IMGN,
    'IMGN_R': IMGN_R,
    'iNaturalist': iNaturalist,
}


def create(name, options):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](options=options)
