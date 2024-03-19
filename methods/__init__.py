from .baseline import BaselineNet
from .core import MultiBranchNet

def get_model(args):
    if args['single_b']:
        net = BaselineNet(args)
    else:
        net = MultiBranchNet(args)
    return net
