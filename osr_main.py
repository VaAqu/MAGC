import os
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

import utils
from steps import *
from methods import get_model
from utils.split import splits_AUROC, splits_F1

import warnings
warnings.filterwarnings('ignore')

def getLoader(options):
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
        Data = Tiny_ImageNet_OSR(known=options['known'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        train_loader, test_loader, out_loader = Data.train_loader, Data.test_loader, Data.out_loader
      
    options['num_known'] = Data.num_known
    return train_loader, test_loader, out_loader
 
 
def main(options):
    if options['single_b']:
        # keys during training
        options['loss_keys']       = ['loss']
        options['acc_keys']        = ['acc']
        # keys during test
        options['test_f1_keys']    = ['f1']
        options['test_acc_keys']   = ['acc']
        options['test_auroc_keys'] = ['auroc']
    else:
        options['loss_keys']       = ['b1', 'b2', 'b3', 'gate_loss', 'total_loss']        
        options['acc_keys']        = ['acc1', 'acc2', 'acc3', 'accGate']
        options['test_f1_keys']    = ['f1', 'f2', 'f3', 'fGate']
        options['test_acc_keys']   = ['tacc1', 'tacc2', 'tacc3', 'taccGate']
        options['test_auroc_keys'] = ['auroc1', 'auroc2', 'auroc3', 'aurocGate']
        
    if options['split'] == 'AUROC':
        splits = splits_AUROC
    elif options['split'] == 'F1':
        splits = splits_F1
    else:
        raise NotImplementedError()
    
    now_time = datetime.datetime.now().strftime("%m%d_%H:%M")
    log_path = './logs/osr' + '/' + options['dataset'] + '/'
    ensure_dir(log_path)
    
    if options['dataset'] == 'cifar100':
        stats_log = open(log_path + "MAGC" + '_' + str(options['plus_num']) + '_' + now_time + '.txt', 'w')
    else:
        stats_log = open(log_path + "MAGC" + '_' + now_time + '.txt', 'w')
    final_result = np.zeros(8)
    for i in range(len(splits[options['dataset']])):
        options['item'] = i
        known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset'] + '-' + str(options['plus_num'])][len(splits[options['dataset']]) - i - 1]
        elif options['dataset'] == 'tiny_imagenet':
            unknown = list(set(list(range(0, 200))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))
        options.update({'known': known, 'unknown': unknown})
        temp_result = trainLoop(options)
        final_result += temp_result
        stats_log.write("SPLIT[%d|5] => Accuracy: [%.3f], AUROC: [%.3f], AUPR_IN: [%.3f], AUPR_OUT: [%.3f], F1-score: [%.3f], Det_Acc: [%.3f]\n" 
                        % (i+1, temp_result[0], temp_result[1], temp_result[2], temp_result[3], temp_result[4], temp_result[5]))
        stats_log.flush()
    stats_log.close()


def trainLoop(options):
    
    # - DATASET -
    train_loader, test_loader, out_loader = getLoader(options)
    
    # - MODEL -
    now_time = datetime.datetime.now().strftime("%m%d_%H:%M")
    ckpt_path = './ckpt/osr' + '/' + options['dataset'] + '/' + now_time
    ensure_dir(ckpt_path)
    model = get_model(options)
    model = nn.DataParallel(model).cuda()

    if options['resume']:
        load_checkpoint(model, options['ckpt'])

    # - OPTIMIZER - 
    extractor_params = model.module.get_params(prefix='extractor')
    classifier_params = model.module.get_params(prefix='classifier')
    lr_cls = options['lr']
    if options['pretrained']:
        lr_extractor = lr_cls*0.1
    else:
        lr_extractor = lr_cls
    params = [
        {'params': classifier_params, 'lr': lr_cls},
        {'params': extractor_params,  'lr': lr_extractor}
    ]
    optimizer = torch.optim.SGD(params, lr=options['lr'], momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options['milestones'], gamma=options['gamma'])

    # - CRITERION - 
    entropy_loss = nn.CrossEntropyLoss().cuda()
    criterion = {'entropy': entropy_loss}

    epoch_start = 0
    if options['resume']:
        checkpoint_dict = load_checkpoint(model, options['ckpt'])
        epoch_start = checkpoint_dict['epoch']
        print(f'== Resuming training process from epoch {epoch_start} >')
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])

    # - MAIN LOOP -
    for epoch in range(epoch_start, options['epoch_num']):
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch: [{epoch+1:d} | {options['epoch_num']:d}] LR: {lr:f}")
        train_loss = train(train_loader, model, criterion, optimizer, args=options)
        if (epoch + 1) % options['test_step'] == 0:
            result_list = evaluation(model, test_loader, out_loader, **options)
        scheduler.step()
        
        # - SAVE CHECKPOINT -
        if (epoch + 1) % options['save_step'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, checkpoint=ckpt_path, filename=f"epoch_{epoch+1}.pth")
            if (epoch + 1) != options['save_step']:
                last_log_path=f"{ckpt_path}/epoch_{epoch+1-options['save_step']}.pth"
                if(os.path.exists(last_log_path)):
                    os.remove(last_log_path)
    
    result_list = evaluation(model, test_loader, out_loader, **options)    
    print("\D-O-N-E!/ =>\nLast ACC:", result_list[0], " Last AUROC:", result_list[1]," Last F1-score:", result_list[4])
    return result_list

if __name__ == '__main__':
    cudnn.benchmark = True
    options = utils.get_config()
    utils.set_seeding(options['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu_ids']
    main(options)
    