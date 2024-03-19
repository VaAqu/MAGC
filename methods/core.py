import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import build_backbone, conv1x1
from utils.misc import cam_dot, cam_mask


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(Classifier, self).__init__()
        self.cls = conv1x1(in_dim, out_dim)


    def forward(self, input):
        logit = self.cls(input)
        if logit.dim() == 1:
            logit =logit.unsqueeze(0)
        return logit
    
    
class MultiBranchNet(nn.Module):
    def __init__(self, args=None):
        super(MultiBranchNet, self).__init__()
        backbone, feature_dim, self.cam_size = build_backbone(img_size=args['img_size'],
                                                              projection_dim=-1, 
                                                              inchan=3)
        self.num_known = args['num_known']
        self.img_size  = args['img_size']
        self.bbox_thr  = args['bbox_thr']
        self.save_pic  = args['save_pic']
        self.ft_list   = {i:[] for i in range(self.num_known)}
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)        
        self.shared_l3 = nn.Sequential(*list(backbone.children())[:-4])
        
        # - BRANCH 1 -
        self.branch1_l4  = nn.Sequential(*list(backbone.children())[-4:-3])
        self.branch1_l5  = nn.Sequential(*list(backbone.children())[-3])
        self.branch1_cls = Classifier(feature_dim, self.num_known, bias=True)

        # - BRANCH 2 -
        self.branch2_l4  = copy.deepcopy(self.branch1_l4)
        self.branch2_l5  = copy.deepcopy(self.branch1_l5)
        self.branch2_cls = Classifier(feature_dim, self.num_known, bias=True)
        
        # - BRANCH 3 -
        self.branch3_l4  = copy.deepcopy(self.branch1_l4)
        self.branch3_l5  = copy.deepcopy(self.branch1_l5)
        self.branch3_cls = Classifier(feature_dim, self.num_known, bias=True)

        self.mix_exps_cls  = Classifier(3*(feature_dim), self.num_known, bias=True)
        
        # - GATING NETWORK - 
        self.gate_l3  = copy.deepcopy(self.shared_l3)
        self.gate_l4  = copy.deepcopy(self.branch1_l4)
        self.gate_l5  = copy.deepcopy(self.branch1_l5)
        self.gate_cls = nn.Sequential(Classifier(feature_dim, int(feature_dim/4), bias=True), Classifier(int(feature_dim/4), 3, bias=True))


    def forward(self, x, y=None):

        b = x.size(0)
        expert1_img = x
            
        # *** BRANCH 1 ***
        feature_b1 = self.shared_l3(expert1_img)
        branch1_l4 = self.branch1_l4(feature_b1)
        branch1_l5 = self.branch1_l5(branch1_l4)
        b1_conv_l5 = self.branch1_cls(branch1_l5)
        b1_logits  = self.avg_pool(b1_conv_l5).view(b, -1)
        b1_cam     = b1_conv_l5.detach().clone()
        if y is not None:
            b1_cam = b1_cam.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b1_cam.shape[-2], b1_cam.shape[-1]))
        else:
            b1_cam = b1_cam.gather(dim=1, index=b1_logits.max[0][1][:,None,None,None].repeat(1, 1, b1_cam.shape[-2], b1_cam.shape[-1]))
        expert2_img = cam_dot(x, branch1_l5, b1_cam, self.img_size, self.cam_size)

        feature_b2 = self.shared_l3(expert2_img.detach())
        branch2_l4 = self.branch2_l4(feature_b2)
        branch2_l5 = self.branch2_l5(branch2_l4)
        b2_conv_l5 = self.branch2_cls(branch2_l5)
        b2_logits  = self.avg_pool(b2_conv_l5).view(b, -1)
        b2_cam     = b2_conv_l5.detach().clone()
        if y is not None:
            b2_cam = b2_cam.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b2_cam.shape[-2], b2_cam.shape[-1]))
        else:
            b2_cam = b2_cam.gather(dim=1, index=b2_cam.max[0][1][:,None,None,None].repeat(1, 1, b2_cam.shape[-2], b2_cam.shape[-1]))
       
        expert3_img = cam_mask(expert1_img, branch2_l5, b2_cam, self.img_size, self.cam_size, self.bbox_thr)
        
        # *** BRANCH 3 ***
        feature_b3 = self.shared_l3(expert3_img.detach())
        branch3_l4 = self.branch3_l4(feature_b3)
        branch3_l5 = self.branch3_l5(branch3_l4)
        b3_pool_l5 = self.avg_pool(branch3_l5).view(b, -1)
        b3_logits  = self.branch3_cls(b3_pool_l5)

        # - GATING NETWORK - 
        gate_l5   = self.gate_l5(self.gate_l4(self.gate_l3(x)))
        gate_pool = self.avg_pool(gate_l5).view(b, -1)
        gate_pred = F.softmax(self.gate_cls(gate_pool)/10, dim=1)

        gate_logits = torch.stack([b1_logits.detach(), b2_logits.detach(), b3_logits.detach()], dim=-1)
        gate_logits = gate_logits * gate_pred.view(gate_pred.size(0), 1, gate_pred.size(1))
        gate_logits = gate_logits.sum(-1)
        
        # - OUTPUT LOGITS -
        logits_list = [b1_logits, b2_logits, b3_logits, gate_logits]
        outputs = {'logits':logits_list, 'gate_pred': gate_pred}        
        return outputs


    def get_params(self, prefix='extractor'):
        extractor_params = list(self.shared_l3.parameters()) +\
                           list(self.branch1_l4.parameters()) + list(self.branch1_l5.parameters()) +\
                           list(self.branch2_l4.parameters()) + list(self.branch2_l5.parameters()) +\
                           list(self.branch3_l4.parameters()) + list(self.branch3_l5.parameters()) +\
                           list(self.gate_l3.parameters()) + list(self.gate_l4.parameters()) + list(self.gate_l5.parameters())
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
        