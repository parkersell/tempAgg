# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from dataset_mistake_detection import SequenceDataset
from os.path import join
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import ValueMeter, calculate_metrics, BalancedSoftmax
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from network import Network
from torch.optim import lr_scheduler
from torch import nn
import copy
from typing import List
import time
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

COMP_PATH = '/data/parkersell/tempAgg/'

pd.options.display.float_format = '{:05.2f}'.format

parser = ArgumentParser(description="Training for Mistake Detection")
parser.add_argument('--view', type=str, default='C10119_rgb')
parser.add_argument("-g", "--gpu", default=0, type=int, help="specify visible devices")
parser.add_argument('--path_to_data', type=str, default=COMP_PATH + 'data/')
parser.add_argument('--path_to_annots', type=str, default=COMP_PATH + 'data/annots/')
parser.add_argument('--path_to_info', type=str, default=COMP_PATH + 'data/infoTempAgg.json')
parser.add_argument('--path_to_models', type=str, default=COMP_PATH + 'models_mistake_detection/',
                    help="Path to the directory where to save all models")

parser.add_argument('--json_directory', type=str, default=COMP_PATH + 'models_mistake_detection/',
                    help='Directory in which to save the generated jsons.')

parser.add_argument('--task', type=str, default='online',
                    choices=['online', 'offline'],
                    help='Task to tackle: online or offline') # offline is semi offline and takes same spanning timestamp as left side on right side of coarse segment

parser.add_argument('--img_tmpl', type=str, default='frame_{:010d}.jpg',
                    help='Template to use to load the representation of a given frame')
parser.add_argument('--resume', action='store_true', help='Whether to resume suspended training')


parser.add_argument('--num_workers', type=int, default=0, help="Number of parallel thread to fetch the data")
parser.add_argument('--display_every', type=int, default=10, help="Display every n iterations")

parser.add_argument('--schedule_on', type=int, default=1, help='')
parser.add_argument('--schedule_epoch', type=int, default=10, help='')

parser.add_argument('--num_coarse_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--num_fine_classes', type=int, default=6, help='Number of classes')
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--latent_dim', type=int, default=512, help='')
parser.add_argument('--linear_dim', type=int, default=512, help='')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='')
parser.add_argument('--scale_factor', type=float, default=-.5, help='')
parser.add_argument('--scale', type=bool, default=True, help='')
parser.add_argument('--batch_size', type=int, default=12, help="Batch Size")
parser.add_argument('--epochs', type=int, default=15, help="Training epochs")
parser.add_argument('--video_feat_dim', type=int, default=2048, choices=[2048, 1024], help='')
parser.add_argument('--past_attention', type=bool, default=True, help='')

# Spanning snippets
parser.add_argument('--spanning_sec', type=float, default=60, help='')
parser.add_argument('--span_dim1', type=int, default=5, help='')
parser.add_argument('--span_dim2', type=int, default=3, help='')
parser.add_argument('--span_dim3', type=int, default=2, help='')

# Recent snippets
parser.add_argument('--recent_dim', type=int, default=5, help='')
parser.add_argument('--recent_sec1', type=float, default=3.0, help='')
parser.add_argument('--recent_sec2', type=float, default=2.0, help='')
parser.add_argument('--recent_sec3', type=float, default=1.0, help='')
parser.add_argument('--recent_sec4', type=float, default=0.0, help='')

# Debugging True
parser.add_argument('--debug_on', type=bool, default=False, help='')

args = parser.parse_args()


def make_model_name(arg_save):

    save_name = "anti_span_{}_s1_{}_s2_{}_s3_{}_recent_{}_r1_{}_r2_{}_r3_{}_r4_{}_bs_{}_drop_{}_lr_{}_dimLa_{}_" \
                "dimLi_{}_epoc_{}_task_{}".format(arg_save.spanning_sec, arg_save.span_dim1,
                                          arg_save.span_dim2, arg_save.span_dim3, arg_save.recent_dim,
                                          arg_save.recent_sec1, arg_save.recent_sec2, arg_save.recent_sec3,
                                          arg_save.recent_sec4, arg_save.batch_size, arg_save.dropout_rate, arg_save.lr,
                                          arg_save.latent_dim, arg_save.linear_dim, arg_save.epochs, args.task)
    
    # Base name without try number
    base_save_name = save_name
    # check if name exists in path_to_models and add try_{#} to name
    try_number = 1
    full_filename = base_save_name + '.txt'
    print(full_filename)
    print(os.listdir(arg_save.path_to_models))

    while full_filename in os.listdir(arg_save.path_to_models):
        save_name = base_save_name + '_try_' + str(try_number) 
        full_filename = save_name + '.txt'
        try_number += 1
    return save_name


def save_model(model, epoch):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, join(args.path_to_models, exp_name + '.pth.tar'))


def log_result(log:List[str], result, type):

    
    log.append("perclip {} Accuracy: {:.5f} ".format(type, result["Acc"]))
    log.append(
        "perclip "+ type +" Recall perclass: \n  - " +
        "\n  - ".join([
            "{}: {:.5f}".format(k, v)
            for k, v in result["per_class_R"].items()
        ])
    )
    log.append(
        "perclip "+ type +" Precision perclass: \n  - " +
        "\n  - ".join([
            "{}: {:.5f}".format(k, v)
            for k, v in result["per_class_P"].items()
        ])
    )

    log.append("---------------------------------------")


def train_test(model, loaders, optimizer, epochs, start_epoch, schedule_on):
    """Training/Test code"""

    loss_mistake_TAB1 = BalancedSoftmax(args.train_num_samples_per_class)
    loss_mistake_TAB2 = BalancedSoftmax(args.train_num_samples_per_class)
    loss_mistake_TAB3 = BalancedSoftmax(args.train_num_samples_per_class)
    loss_mistake_TAB4 = BalancedSoftmax(args.train_num_samples_per_class)

    start = time.time()

    for epoch in range(start_epoch, epochs):
        

        # define training and test meters
        mistake_loss_meter = {'train': ValueMeter(), 'test': ValueMeter()}

        output_keys = ['mistake1', 'mistake2', 'mistake3', 'mistake4', 'ensemble']
        preds = {key: [] for key in output_keys}
        gt_labels = []
        

        for mode in ['train', 'test']:

            # enable gradients only if train
            with torch.set_grad_enabled(mode == 'train'):
                if mode == 'train':
                    model.train()
                else:
                    model.eval()

                pbar = tqdm(
                    loaders[mode],
                    desc="{}ing epoch {}".format(mode.capitalize(), epoch)
                )    

                for i, batch in enumerate(pbar):
                    x_spanning = batch['spanning_features']
                    x_recent = batch['recent_features']
                    if type(x_spanning) == list:
                        x_spanning = [xx.to(device) for xx in x_spanning]
                        x_recent = [xx.to(device) for xx in x_recent]
                    else:
                        x_spanning = x_spanning.to(device)
                        x_recent = x_recent.to(device)

                    y_label = batch['label'].to(device).squeeze()
                    bs = y_label.shape[0]  # batch size

                    pred_mistake1, pred_mistake2, pred_mistake3, pred_mistake4 = model(x_spanning, x_recent)


                    
                    loss = loss_mistake_TAB1(pred_mistake1, y_label) + \
                           loss_mistake_TAB2(pred_mistake2, y_label) + \
                           loss_mistake_TAB3(pred_mistake3, y_label) + \
                           loss_mistake_TAB4(pred_mistake4, y_label)
                    mistake_loss_meter[mode].add(loss.item(), bs)

                    # Output log for current batch
                    pbar.set_postfix({
                        "loss": "{:.5f}".format(loss.item()),
                        "lr": "{:.5f}".format(optimizer.param_groups[0]['lr']),
                        "mode": mode,
                        "sched_lr": schedule_on.get_last_lr() if schedule_on is not None else "None"
                    })
                    
                    # store the values in the meters to keep incremental averages
                    # if in train mode
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        
                    else: 
                        gt_labels.extend(y_label.detach().cpu().tolist())
                        preds['mistake1'].extend(pred_mistake1.argmax(1).detach().cpu().tolist())
                        preds['mistake2'].extend(pred_mistake2.argmax(1).detach().cpu().tolist())
                        preds['mistake3'].extend(pred_mistake3.argmax(1).detach().cpu().tolist())
                        preds['mistake4'].extend(pred_mistake4.argmax(1).detach().cpu().tolist())
                        preds['ensemble'].extend((pred_mistake1.detach() + pred_mistake2.detach() + pred_mistake3.detach() + pred_mistake4.detach()).argmax(1).cpu().tolist())
                        # with open(args.path_to_models + '/' + exp_name + '_preds.txt', 'a') as f:
                        #     f.write("===================================================================\nLABEL: ")
                        #     f.write(','.join(list(map(str, y_label.detach().cpu().tolist()))))
                        #     f.write("\n-------------------------------------------------------------------\nMIST1: ")
                        #     f.write(','.join(list(map(str, pred_mistake1.argmax(1).detach().cpu().tolist()))))
                        #     f.write("\n-------------------------------------------------------------------\nMIST2: ")
                        #     f.write(','.join(list(map(str, pred_mistake2.argmax(1).detach().cpu().tolist()))))
                        #     f.write("\n-------------------------------------------------------------------\nMIST3: ")
                        #     f.write(','.join(list(map(str, pred_mistake3.argmax(1).detach().cpu().tolist()))))
                        #     f.write("\n-------------------------------------------------------------------\nMIST4: ")
                        #     f.write(','.join(list(map(str, pred_mistake4.argmax(1).detach().cpu().tolist()))))
                        #     f.write("\n-------------------------------------------------------------------\nENS:   ")
                        #     f.write(','.join(list(map(str, (pred_mistake1.detach() + pred_mistake2.detach() + pred_mistake3.detach() + pred_mistake4.detach()).argmax(1).cpu().tolist()))))
                        #     f.write("\n-------------------------------------------------------------------\n")
                        #     f.write(str(loss.item()))
                        #     f.write(' ')
                        #     f.write(str(schedule_on.get_last_lr()))
                        #     f.write(' ')
                        #     f.write(str(bs))
                        #     f.write(' ')
                        #     f.write(str(i))
                        #     f.write("\n===================================================================\n\n")
        if schedule_on is not None:
            schedule_on.step()

        end = time.time()

        # Output log for current epoch
        log = []
        log.append(f"Epoch {epoch}")
        log.append("---------------------------------------")
        log.append("train loss: {:.5f}".format(mistake_loss_meter['train'].value()))

        log.append("test loss: {:.5f}".format(mistake_loss_meter['test'].value()))
        log.append("---------------------------------------")

        log.append("running time: {:.2f} sec".format(end - start))

        for key in output_keys:
            log_result(log, calculate_metrics(gt_labels, preds[key], args.coarse_class_names), key)

        log.append("=======================================")
        print("\n".join(log))
        log.append("=======================================\n")
        print("=======================================")



        # save checkpoint at the end of each train/test epoch
        save_model(model, epoch + 1)

        with open(args.path_to_models + '/' + exp_name + '.txt', 'a') as f:
            f.write("\n".join(log))


def load_checkpoint(model):
    model_add = '.pth.tar'

    chk = torch.load(join(args.path_to_models, exp_name + model_add))
    epoch = chk['epoch']
    model.load_state_dict(chk['state_dict'])
    return epoch


def get_loader(mode):
    path_to_lmdb = join(args.path_to_data, 'TSM_features', args.view)

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_annots': args.path_to_annots,
        'path_to_info': args.path_to_info,
        'mode': mode,
        'args': args
    }
    _set = SequenceDataset(**kargs)
    if mode == 'train':
        args.train_num_samples_per_class = _set.num_samples_per_class

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=True)


def get_model():
    return Network(args)
        



def main():
    model = get_model()
    model.to(device)
    torch.cuda.set_device(args.gpu)

    loaders = {m: get_loader(m) for m in ['train', 'test']}

    if args.resume:
        start_epoch  = load_checkpoint(model)
    else:
        start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    schedule_on = None
    if args.schedule_on:
        schedule_on = lr_scheduler.StepLR(optimizer, args.schedule_epoch, gamma=0.1, last_epoch=-1)

    train_test(model, loaders, optimizer, args.epochs, 
                        start_epoch, schedule_on)




if __name__ == '__main__':

    assert args.json_directory is not None

    exp_name = make_model_name(args)
    print("Save file name ", exp_name)
    print("Printing Arguments ")
    print(args)
    with open(args.path_to_info, "r") as f:
        data_info = json.load(f)
    args.coarse_class_names = data_info["coarse"]["class_names"]
    args.fine_class_names = data_info["fine"]["class_names"]

    main()