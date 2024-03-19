import time
import torch
import torch.nn.functional as F

from utils import *

def exclude_gt(logit, target, is_log=False):
    logit = F.log_softmax(logit, dim=-1) if is_log else F.softmax(logit, dim=-1)
    mask = torch.ones_like(logit)
    for i in range(logit.size(0)): 
        mask[i, target[i].long()] = 0

    return mask*logit


def train(train_loader, model, criterion, optimizer, args):
    # - ENTER TRAINING MODE -
    model.train()
    # - UPDATE FOR EVERY EXPERT -
    loss_keys = args['loss_keys']
    acc_keys  = args['acc_keys']
    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter  = {p: AverageMeter() for p in acc_keys}

    time_start = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        inputs = inputs.cuda()
        target = target.cuda()
        output_dict = model(inputs, target)
        logits = output_dict['logits']
        loss_values = [criterion['entropy'](logit.float(), target.long()) for k, logit in enumerate(logits)]
        loss_values.append(args['loss_wgts'][0] * loss_values[0] +\
                           args['loss_wgts'][1] * loss_values[1] +\
                           args['loss_wgts'][2] * loss_values[2] +\
                           args['loss_wgts'][3] * loss_values[3])

        loss_content = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}
        acc_values = [accuracy(logit, target, topk = (1,))[0] for logit in logits]
        acc_content = {acc_keys[k] : acc_values[k] for k in range(len(acc_keys))}
        update_meter(loss_meter, loss_content, inputs.size(0))
        update_meter(acc_meter, acc_content, inputs.size(0))

        tmp_str = "< Training Loss >\n"
        i = 0
        for k, v in loss_meter.items(): 
            if i == 9:
                tmp_str = tmp_str
                temp = f"{k}:{v.value:.4f} "
            elif i == 12:
                tmp_str = tmp_str + "\n" + temp + f"{k}:{v.value:.4f} "
            else:
                tmp_str = tmp_str + f"{k}:{v.value:.4f} "
            i += 1
        tmp_str = tmp_str + "\n< Training Accuracy >"
        i = 0
        for k, v in acc_meter.items():
            if i % 9 == 0:
                tmp_str = tmp_str + "\n"
            i += 1
            tmp_str = tmp_str + f"{k}:{v.value:.1f} "
        optimizer.zero_grad()
        loss_values[-1].backward()
        optimizer.step()
    
    time_eclapse = time.time() - time_start
    print(tmp_str + f"t:{time_eclapse:.1f}s")
    
    return loss_meter[loss_keys[-1]].value
