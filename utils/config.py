import os
import yaml
import argparse
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def parser2dict():
    config, _ = parser.parse_known_args()
    option = edict(config.__dict__)
    
    return edict(option)


def _merge_a_into_b(a, b):
    
    """
    Merge config dictionary a into b,
    clobbering the options in b whenever they are also specified in a.
    """

    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b)
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def print_options(args):
    
        """
        Print and save options:     
        1. Print both current options and default values(if different).
        2. Save options into a text file / [checkpoints_dir] / args.txt.
        """
        
        message = ''
        message += ' < Options > \n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:<15}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += ' <  End  >\n'
        print(message)


def cfg_from_file(option):
    
    """Load a config from file <filename> and merge it into the default options."""

    filename=option.config
    # args from yaml file
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))
    _merge_a_into_b(yaml_cfg, option)

    return option


def get_config():
    option = parser2dict()
    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.environ['POSE_PARAM_PATH'] + '/' + filename
    option = cfg_from_file(option)
    print_options(option)
    return option


# parameter list
# base
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='batch size (default: 128)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model')
parser.add_argument('-j', '--num_workers', default=2, type=int, metavar='N', help='data loading workers')
parser.add_argument('-g', '--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--seed', default=71324, type=int, help='DO NOT MODIFIED IT QWQ')
parser.add_argument('--config', default='config/train.yml', type=str, help='config files')
parser.add_argument('--pretrained', default=False, type=bool, help='using pretrained model (default: none)')
parser.add_argument('--epoch_num', default=100, type=int, metavar='N', help='number of total epoch_num to run')
parser.add_argument('--resume', default=None, type=str, metavar='path', help='path to latest checkpoint (default: none)')

# test
parser.add_argument('--num_sample', default=1, type=int, metavar='N', help='number of sample')
