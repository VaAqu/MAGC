import os
import wandb
import torch
import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datasets.ood_loader as ood_loader

import utils
from steps import *
from methods import get_model

import warnings
warnings.filterwarnings('ignore')

def main(options):
    if options['single_b']:
        # keys during training
        options['acc_keys']        = ['acc']
        options['loss_keys']       = ['loss']
        # keys during test
        options['test_f1_keys']    = ['f1']
        options['test_acc_keys']   = ['acc']
        options['test_auroc_keys'] = ['auroc']
    else:
        options['loss_keys']       = ['b1', 'b2', 'b3', 'gate_loss', 'total_loss']        
        options['acc_keys']        = ['acc1', 'acc2', 'acc3', 'gate_acc']
        options['test_f1_keys']    = ['f1', 'f2', 'f3', 'gate_f']
        options['test_acc_keys']   = ['tacc1', 'tacc2', 'tacc3', 'gate_tacc']
        options['test_auroc_keys'] = ['auroc1', 'auroc2', 'auroc3', 'gate_auroc']
        
    
    file_name = './logs/' + '/' + options['id_set'] + '/'
    utils.ensure_dir(file_name)
    stats_log = open(file_name + "GProject_ood" + '.txt', 'w')

    # wandb record
    if options['use_wandb']:
        now_time = (datetime.datetime.now()).strftime("%m%d_%H:%M")
        wandb.init(name = now_time + '_' + 'loss_w: ' + str(options['loss_weights']) + ' score_w: ' + str(options['score_weights']), project = "Graduation_Project_" + options['id_set'], reinit = False)
    
    # main train loop
    temp_result = train_loop(options)
    stats_log.write("< Accuracy: [%.3f], AUROC: [%.3f], AUPR_IN: [%.3f], AUPR_OUT: [%.3f], F1-score: [%.3f], Det_Acc: [%.3f] >\n" % (temp_result[0], temp_result[1], temp_result[2], temp_result[3], temp_result[4], temp_result[5]))
    stats_log.write("** DONE **\n")
    stats_log.flush()
    stats_log.close()
        
def data_prepare(options):
    print("ID Set:{} & OOD Set:{} Preparation".format(options['id_set'], options['ood_set']))
    
    id_dataset  = ood_loader.create(options['id_set'], options=options)
    out_dataset = ood_loader.create(options['ood_set'], options=options)
    train_loader, test_loader = id_dataset.train_loader, id_dataset.test_loader
    out_loader = out_dataset.test_loader
    
    return train_loader, test_loader, out_loader

def train_loop(options):
    
    # - DATASET -
    train_loader, test_loader, out_loader = data_prepare(options)
    
    # - MODEL -
    utils.ensure_dir(options['ckpt'])
    model = get_model(options)
    model = nn.DataParallel(model).cuda()

    if options['resume']:
        utils.load_checkpoint(model, options['ckpt'])

    # - OPTIMIZER - 
    extractor_params = model.module.get_params(prefix='extractor')
    classifier_params = model.module.get_params(prefix='classifier')
    lr_cls = options['lr']
    lr_extractor = 0.1 * lr_cls
    params = [
        {'params': classifier_params, 'lr': lr_cls},
        {'params': extractor_params, 'lr': lr_extractor}
    ]
    if options['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, lr=options['lr'], momentum=options['momentum'], weight_decay=options['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options['milestones'], gamma=options['gamma'])
    elif options['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=options['lr'], betas=(0.9,0.99), weight_decay=options['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options['milestones'], gamma=options['gamma'])
    else:
        optimizer = torch.optim.Adam(params, lr=options['lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options['epoch_num'] // 2, gamma=options['gamma'])

    # - CRITERION - 
    entropy_loss = nn.CrossEntropyLoss().cuda()
    criterion = {'entropy': entropy_loss}

    epoch_start = 0
    if options['resume']:
        checkpoint_dict = utils.load_checkpoint(model, options['ckpt'])
        epoch_start = checkpoint_dict['epoch']
        print(f'== Resuming training process from epoch {epoch_start} >')
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])

    # - MAIN LOOP -
    for epoch in range(epoch_start, options['epoch_num']):
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch: [{epoch+1:d} | {options['epoch_num']:d}] LR: {lr:f}")
        train_loss = train(train_loader, model, criterion, optimizer, args=options)
        if options['use_wandb']:
            wandb.log({"Train Loss" : train_loss})
        if (epoch + 1) % options['test_step'] == 0:
            result_list = evaluation(model, test_loader, out_loader, **options)
            if options['use_wandb']:
                wandb.log({
                    "Test OSCR"     : result_list[7],                   
                    "Test Macro-F1" : result_list[4],
                    "Test AUPR_out" : result_list[3],
                    "Test AUPR_in"  : result_list[2],
                    "Test AUROC"    : result_list[1],
                    "Test Accuracy" : result_list[0]
                })        
        scheduler.step()
        
        # - SAVE CHECKPOINT -
        if (epoch + 1) % options['save_step'] == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, checkpoint=options['ckpt'], filename=f"epoch_{epoch+1}.pth")
            if (epoch + 1) != options['save_step']:
                last_file_name=f"{options['ckpt']}/epoch_{epoch+1-options['save_step']}.pth"
                if(os.path.exists(last_file_name)):
                    os.remove(last_file_name)
    
    result_list = evaluation(model, test_loader, out_loader, **options)    
    print("\D-O-N-E!/ =>\nLast ACC:", result_list[0], " Last AUROC:", result_list[1]," Last F1-score:", result_list[4])
    return result_list


if __name__ == '__main__':
    cudnn.benchmark = True
    options = utils.get_config()
    utils.set_seeding(options['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu_ids']
    main(options)
    