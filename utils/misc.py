import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from sklearn.metrics import accuracy_score
from datasets.osr_loader import CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(dir_name):
    
    """ check if dir exist, if not create new folder """
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth'):
    
    """ save checkpoint """
    
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def update_meter(dict_meter, dict_content, batch_size):
    
    """ update meter """
    
    for key, value in dict_meter.items():
        if isinstance(dict_content[key], torch.Tensor):
            value.update(dict_content[key].item(), batch_size)
        else:
            value.update(dict_content[key], batch_size)

def load_checkpoint(model, pth_file):
        
    """ load checkpoint """
    
    print(' < Reading from Checkpoint > ')
    # assert os.path.isfile(pth_file), 'Error: No Checkpoint Directory Found!'
    checkpoint = torch.load(pth_file)
    pretrained_dict = checkpoint['state_dict']
    
    model_dict = model.module.state_dict()
    model_dict.update(pretrained_dict)
    model.module.load_state_dict(model_dict)
    print(" < Loading from Checkpoint: '{}' (epoch {}) > ".format(pth_file, checkpoint['epoch']))

    return checkpoint 

class AverageMeter(object):
    
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.avg


class ComputeAcc:
    
    """ compute more accurate ACC """
    
    def __init__(self):
        self.avg = 0.
        self.pred = []
        self.tgt = []

    def update(self, logits, tgt):
        pred = torch.argmax(logits, dim=1)
        self.pred.extend(pred.cpu().numpy().tolist())
        self.tgt.extend(tgt.cpu().numpy().tolist())

    def __str__(self):
        return str(self.value)
    
    @property
    def value(self):
        self.avg = accuracy_score(self.pred, self.tgt)
        return self.avg*100

def set_seeding(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # cpu vars
    torch.cuda.manual_seed_all(seed)   # gpu vars
    
    cudnn.benchmark = True
    cudnn.deterministic = True

def init_params(net):
    
    ''' init layer parameters '''
    
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cam_dot(x, conv5_cam, img_size, cam_size):
    
    mask_sum  = conv5_cam.sum(1, keepdim=True).view(conv5_cam.size(0), 1, cam_size, cam_size)
    mask_sum  = F.interpolate(mask_sum, size=(img_size, img_size), mode='bilinear', align_corners=True)
    mask_sum  = mask_sum.view(mask_sum.size(0), -1)
    cam_mean  = mask_sum.mean(-1, keepdim=True)
    cam_std   = mask_sum.std(-1, keepdim=True)
    cam_map   = ((mask_sum - cam_mean) / cam_std) * 0.5 + 1
    cam_map   = cam_map.view(cam_map.size(0), 1, img_size, img_size)
    
    sub_expert_img = x * cam_map

    return sub_expert_img


def cam_mask(x, conv5_cam, img_size, cam_size, rate=0.1):

    mask_sum  = conv5_cam.sum(1, keepdim=True).view(conv5_cam.size(0), 1, cam_size, cam_size)
    mask_sum  = F.interpolate(mask_sum, size=(img_size, img_size), mode='bilinear', align_corners=True)
    mask_sum  = mask_sum.view(mask_sum.size(0), -1)
    x_range   = mask_sum.max(-1, keepdim=True)[0] - mask_sum.min(-1, keepdim=True)[0]
    mask_sum  = (mask_sum - mask_sum.min(-1, keepdim=True)[0])/x_range
    rate_high = torch.topk(mask_sum, k = int(rate*mask_sum.shape[-1]), dim=-1)[0][:,-1]
    rate_low  = torch.topk(mask_sum, k = int(rate*mask_sum.shape[-1]), largest=False, dim=-1)[0][:,-1]
    
    mask_high = mask_sum < rate_high.view(rate_high.size(0), 1)
    mask_low  = mask_sum > rate_low.view(rate_high.size(0), 1)
    mask_fin  = (mask_high*mask_low).view(mask_low.size(0), 1, img_size, img_size)
    
    sub_expert_img = torch.zeros_like(x)
    sub_expert_img = x * mask_fin

    return sub_expert_img


def data_prepare(options):
    print("{} Preparation".format(options['dataset']))
    if 'cifar10' == options['dataset']:
        options['img_size'] = 32
        Data = CIFAR10_OSR(known=options['known'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        train_loader, test_loader, out_loader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        options['img_size'] = 32
        Data = SVHN_OSR(known=options['known'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        train_loader, test_loader, out_loader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        options['img_size'] = 32
        Data = CIFAR10_OSR(known=options['known'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        train_loader, test_loader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        out_loader = out_Data.test_loader
    elif 'tiny_imagenet' in options['dataset']:
        options['img_size'] = 64
        options['cam_size'] = 8
        Data = Tiny_ImageNet_OSR(known=options['known'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        train_loader, test_loader, out_loader = Data.train_loader, Data.test_loader, Data.out_loader
    options['num_known'] = Data.num_known
    
    return train_loader, test_loader, out_loader


splits_F1 = {
    'mnist': [
        [3, 7, 8, 2, 4, 6],
        [7, 1, 0, 9, 4, 6],
        [8, 1, 6, 7, 2, 4],
        [7, 3, 8, 4, 6, 1],
        [2, 8, 7, 3, 5, 1]
    ],
    'svhn': [
        [3, 7, 8, 2, 4, 6],
        [7, 1, 0, 9, 4, 6],
        [8, 1, 6, 7, 2, 4],
        [7, 3, 8, 4, 6, 1],
        [2, 8, 7, 3, 5, 1]
    ],
    'cifar10': [
        [3, 7, 8, 2, 4, 6],
        [7, 1, 0, 9, 4, 6],
        [8, 1, 6, 7, 2, 4],
        [7, 3, 8, 4, 6, 1],
        [2, 8, 7, 3, 5, 1]
    ],
    'cifar100': [
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9]
    ],
    'cifar100-10': [
        [27, 46, 98, 38, 72, 31, 36, 66, 3, 97],
        [98, 46, 14, 1, 7, 73, 3, 79, 93, 11],
        [79, 98, 67, 7, 77, 42, 36, 65, 26, 64],
        [46, 77, 29, 24, 65, 66, 79, 21, 1, 95],
        [21, 95, 64, 55, 50, 24, 93, 75, 27, 36]
    ],
    'cifar100-50': [
        [27, 46, 98, 38, 72, 31, 36, 66, 3, 97, 75, 67, 42, 32, 14, 93, 6, 88, 11, 1, 44,
        35, 73, 19, 18, 78, 15, 4, 50, 65, 64, 55, 30, 80, 26, 2, 7, 34, 79, 43, 74, 29,
        45, 91, 37, 99, 95, 63, 24, 21],
        [98, 46, 14, 1, 7, 73, 3, 79, 93, 11, 37, 29, 2, 74, 91, 77, 55, 50, 18, 80, 63,
        67, 4, 45, 95, 30, 75, 97, 88, 36, 31, 27, 65, 32, 43, 72, 6, 26, 15, 42, 19,
        34, 38, 66, 35, 21, 24, 99, 78, 44],
        [79, 98, 67, 7, 77, 42, 36, 65, 26, 64, 66, 73, 75, 3, 32, 14, 35, 6, 24, 21, 55,
        34, 30, 43, 93, 38, 19, 99, 72, 97, 78, 18, 31, 63, 29, 74, 91, 4, 27, 46, 2, 88,
        45, 15, 11, 1, 95, 50, 80, 44],
        [46, 77, 29, 24, 65, 66, 79, 21, 1, 95, 36, 88, 27, 99, 67, 19, 75, 42, 2, 73,
        32, 98, 72, 97, 78, 11, 14, 74, 50, 37, 26, 64, 44, 30, 31, 18, 38, 4, 35, 80,
        45, 63, 93, 34, 3, 43, 6, 55, 91, 15],
        [21, 95, 64, 55, 50, 24, 93, 75, 27, 36, 73, 63, 19, 98, 46, 1, 15, 72, 42, 78,
        77, 29, 74, 30, 14, 38, 80, 45, 4, 26, 31, 11, 97, 7, 66, 65, 99, 34, 6, 18, 44,
        3, 35, 88, 43, 91, 32, 67, 37, 79]
    ],
    'tiny_imagenet': [
        [2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193],
        [4, 11, 32, 42, 51, 53, 67, 84, 87, 104, 116, 140, 144, 145, 148, 149, 155, 168, 185, 193],
        [3, 9, 10, 20, 23, 28, 29, 45, 54, 74, 133, 143, 146, 147, 156, 159, 161, 170, 184, 195],
        [1, 15, 17, 31, 36, 44, 66, 69, 84, 89, 102, 137, 154, 160, 170, 177, 182, 185, 195, 197],
        [4, 14, 16, 33, 34, 39, 59, 69, 77, 92, 101, 103, 130, 133, 147, 161, 166, 168, 172, 173]
    ]
}

splits_AUROC = {
    'mnist': [
        [0, 1, 2, 4, 5, 9],
        [0, 3, 5, 7, 8, 9],
        [0, 1, 5, 6, 7, 8],
        [3, 4, 5, 7, 8, 9],
        [0, 1, 2, 3, 7, 8]
    ],
    'svhn': [
        [0, 1, 2, 4, 5, 9],
        [0, 3, 5, 7, 8, 9],
        [0, 1, 5, 6, 7, 8],
        [3, 4, 5, 7, 8, 9],
        [0, 1, 2, 3, 7, 8]
    ],
    'cifar10': [
        [0, 1, 2, 4, 5, 9],
        [0, 3, 5, 7, 8, 9],
        [0, 1, 5, 6, 7, 8],
        [3, 4, 5, 7, 8, 9],
        [0, 1, 2, 3, 7, 8]
    ],
    'cifar100': [
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9]
    ],
    'cifar100-10': [
        [26, 31, 34, 44, 45, 63, 65, 77, 93, 98],
        [7, 11, 66, 75, 77, 93, 95, 97, 98, 99],
        [2, 11, 15, 24, 32, 34, 63, 88, 93, 95],
        [1, 11, 38, 42, 44, 45, 63, 64, 66, 67],
        [3, 15, 19, 21, 42, 46, 66, 72, 78, 98]
    ],
    'cifar100-50': [
        [1, 2, 7, 9, 10, 12, 15, 18, 21, 23, 26, 30, 32, 33, 34, 36, 37, 39, 40, 42, 44, 45, 46, 47, 49, 50, 51, 52, 55,
         56, 59, 60, 61, 63, 65, 66, 70, 72, 73, 74, 76, 78, 80, 83, 87, 91, 92, 96, 98, 99],
        [0, 2, 4, 5, 9, 12, 14, 17, 18, 20, 21, 23, 24, 25, 31, 32, 33, 35, 39, 43, 45, 49, 50, 51, 52, 54, 55, 56, 60,
         64, 65, 66, 68, 70, 71, 73, 74, 77, 78, 79, 80, 82, 83, 86, 91, 93, 94, 96, 97, 98],
        [0, 4, 10, 11, 12, 14, 15, 17, 18, 21, 23, 26, 27, 28, 29, 31, 32, 33, 36, 39, 40, 42, 43, 46, 47, 51, 53, 56, 57,
         59, 60, 64, 66, 71, 73, 74, 75, 76, 78, 79, 80, 83, 87, 91, 92, 93, 94, 95, 96, 99],
        [0, 2, 5, 6, 9, 10, 11, 12, 14, 16, 18, 19, 21, 22, 23, 26, 27, 28, 29, 31, 33, 35, 36, 37, 38, 39, 40, 43, 45,
         49, 52, 56, 59, 61, 62, 63, 64, 65, 71, 74, 75, 78, 80, 82, 86, 87, 91, 93, 94, 96],
        [0, 1, 4, 6, 7, 12, 15, 16, 17, 19, 20, 21, 22, 23, 26, 27, 28, 32, 39, 40, 42, 43, 44, 47, 49, 50, 52, 53, 54,
         55, 56, 59, 61, 62, 63, 65, 66, 67, 68, 73, 74, 77, 82, 83, 86, 87, 93, 94, 97, 98]
    ],
    'tiny_imagenet': [
        [2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193],
        [4, 11, 32, 42, 51, 53, 67, 84, 87, 104, 116, 140, 144, 145, 148, 149, 155, 168, 185, 193],
        [3, 9, 10, 20, 23, 28, 29, 45, 54, 74, 133, 143, 146, 147, 156, 159, 161, 170, 184, 195],
        [1, 15, 17, 31, 36, 44, 66, 69, 84, 89, 102, 137, 154, 160, 170, 177, 182, 185, 195, 197],
        [4, 14, 16, 33, 34, 39, 59, 69, 77, 92, 101, 103, 130, 133, 147, 161, 166, 168, 172, 173]
    ],
    'imagenet_100': [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
         52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
         76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    ]
}