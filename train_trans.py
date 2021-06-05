from models.load_data import TransDataset
from models.vit_models.vit import Count_Vit
from models.vit_models.cross_vit import Count_Cross_Vit
from models.vit_models.trans_loss import CountLoss
from models.mcan.optim import get_optim, adjust_lr

import numpy as np
import datetime
import time
import shutil
import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.cuda.amp import autocast


def parse_args():
    parser = argparse.ArgumentParser(description='VQA MODELS')

    parser.add_argument('run_mode', type=str, help='train or val')
    parser.add_argument('cfg_path', type=str, help='config file path')
    parser.add_argument('version', type=str, help='model version')

    args = parser.parse_args()
    return args


def load_configs(cfg_path):
    with open(cfg_path, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        # cfg = yaml.load(cfg_file)
    # cfg['image_size'] = tuple(cfg.get('image_size'))
    cfg['OPT_BETAS'] = tuple(cfg.get('OPT_BETAS'))

    return cfg


def train(config, dataset):
    # init log
    print('Initializing log file -->')
    if os.path.exists(config['LOG_PATH'] + 'log_run_' + config['version'] + '.txt'):
        os.remove(config['LOG_PATH'] + 'log_run_' + config['version'] + '.txt')
    print('Finished!')
    print('')

    # run train
    data_size = dataset.data_size

    # init model
    if config['model_name'] == 'vit':
        model = Count_Vit(config)
    else:
        model = Count_Cross_Vit(config)
    print(model)
    model.cuda()
    # model.train()

    model = nn.DataParallel(model, device_ids=config['DEVICES'])
    loss_fn = CountLoss(alpha=0.2, beta=0.8)

    if ('ckpt_' + config['version']) in os.listdir(config['CKPTS_PATH']):
        shutil.rmtree(config['CKPTS_PATH'] + 'ckpt_' + config['version'])
    os.mkdir(config['CKPTS_PATH'] + 'ckpt_' + config['version'])

    optim = get_optim(config, model, data_size)
    start_epoch = 0

    loss_sum = 0
    loss_sum_ce = 0
    loss_sum_bce = 0
    named_params = list(model.named_parameters())
    grad_norm = np.zeros(len(named_params))

    # split datasets
    n_train = int(config['train_rate'] * data_size)
    n_val = data_size - n_train
    train_datasets, val_datasets = random_split(
        dataset, [n_train, n_val],
        torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(
        train_datasets,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEM'],
    )

    val_loader = DataLoader(
        val_datasets,
        batch_size=config['EVAL_BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEM'],
    )

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=config['BATCH_SIZE'],
    #     shuffle=True,
    #     num_workers=config['NUM_WORKERS'],
    #     pin_memory=config['PIN_MEM'],
    #     drop_last=True
    # )

    # Training script
    for epoch in range(start_epoch, config['MAX_EPOCH']):

        # Save log information
        logfile = open(
            config['LOG_PATH'] +
            'log_run_' + config['version'] + '.txt',
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()

        # Learning Rate Decay
        if epoch in config['LR_DECAY_LIST']:
            adjust_lr(optim, config['LR_DECAY_R'])

        time_start = time.time()
        # Iteration
        model.train()
        for step, (image_data, num_gt, class_gt) in enumerate(train_loader):
            optim.zero_grad()

            image_data = image_data.cuda()
            num_gt = num_gt.cuda()
            class_gt = class_gt.cuda()

            # with autocast():
            pred_class, pred_num = model(image_data)
            loss, loss_ce, loss_bce = loss_fn(pred_num, pred_class, num_gt, class_gt)
            loss.backward()

            loss_sum += loss.cpu().data.numpy()
            loss_sum_ce += loss_ce.cpu().data.numpy()
            loss_sum_bce += loss_bce.cpu().data.numpy()

            print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                config['version'],
                epoch + 1,
                step,
                int(data_size / config['BATCH_SIZE']),
                config['run_mode'],
                loss.cpu().data.numpy() / config['BATCH_SIZE'],
                optim._rate
            ), end='          ')

            # Gradient norm clipping
            if config['GRAD_NORM_CLIP'] > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['GRAD_NORM_CLIP']
                )

            # Save the gradient information
            for name in range(len(named_params)):
                norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                    if named_params[name][1].grad is not None else 0
                grad_norm[name] += norm_v

            optim.step()

        time_end = time.time()
        print('Finished in {}s'.format(int(time_end - time_start)))
        epoch_finish = epoch + 1

        # Save checkpoint
        if epoch_finish % 50 == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                config['CKPTS_PATH'] +
                'ckpt_' + config['version'] +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

        # Eval after every epoch
        correct = 0.0
        model.eval()
        for step, (image_data, num_gt, _) in enumerate(val_loader):
            image_data = image_data.cuda()
            num_gt = num_gt.cuda()
            _, pred = model(image_data)
            output = pred.data.max(1)[1]
            correct += float(output.eq(num_gt.data).sum())
        acc = correct / n_val

        print("\r[version %s][epoch %2d] acc: %.4f" % (
            config['version'],
            epoch + 1,
            acc
        ), end='          ')

        # Logging
        logfile = open(
            config['LOG_PATH'] +
            'log_run_' + config['version'] + '.txt',
            'a+'
        )
        logfile.write(
            'epoch = ' + str(epoch_finish) +
            '  loss = ' + str(loss_sum / data_size) +
            '  / ' + str(loss_sum_ce / data_size) +
            '  / ' + str(loss_sum_bce / data_size) +
            '\n' +
            'lr = ' + str(optim._rate) +
            '  acc = ' + str(acc) +
            '\n\n'
        )
        logfile.close()

        loss_sum = 0
        loss_sum_ce = 0
        loss_sum_bce = 0
        grad_norm = np.zeros(len(named_params))


def val(config, dataset):
    # Load parameters
    ckpt_path = config['CKPTS_PATH'] + 'ckpt_' + config['version'] + \
           '/epoch' + str(config['CKPT_EPOCH']) + '.pkl'
    print('Loading ckpt {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)['state_dict']
    print('Finish!')

    pred_list = []
    # data_size = dataset.data_size

    if config['model_name'] == 'vit':
        model = Count_Vit(config)
    else:
        model = Count_Cross_Vit(config)
    model.cuda()
    model.eval()
    model = nn.DataParallel(model, device_ids=config['DEVICES'])

    model.load_state_dict(state_dict)

    dataloader = DataLoader(
        dataset,
        batch_size=config['EVAL_BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )

    for step, (image_data, num_gt, class_gt) in enumerate(dataloader):
        # print("\rEvaluation: [step %4d/%4d]" % (
        #     step,
        #     int(data_size / config['EVAL_BATCH_SIZE']),
        # ), end='          ')

        image_data = image_data.cuda()

        pred_class, pred_num = model(image_data)

        pred_class = pred_class.cpu().data.numpy()
        pred_class = np.argmax(pred_class, axis=1)

        print('class_label : ', class_gt.numpy())
        print('predict : ', pred_class)

        pred_num = pred_num.cpu().data.numpy()
        pred_num = np.argmax(pred_num, axis=1)

        for pred in pred_num:
            pred_list.append(pred)

    print('Predict Finished! ')

    pred_list = [str(pred) + '\n' for pred in pred_list]

    with open('results/answers/count_{}.txt'.format(config['version']), 'w') as f:
        for i in pred_list:
            f.write(i)


def run_model():
    args = parse_args()
    config = load_configs(args.cfg_path)
    config['run_mode'] = args.run_mode
    config['version'] = args.version

    if args.run_mode == 'train':
        config['image_path'] += 'Train_Image/'
        config['question_path'] += 'Train_count.json'

        print('Loading datasets -->')
        dataset = TransDataset(config)

        train(config, dataset)

    else:
        config['image_path'] += 'Valid_Image/'
        config['question_path'] += 'Valid_count.json'

        print('Loading datasets -->')
        dataset = TransDataset(config)

        val(config, dataset)


if __name__ == '__main__':
    run_model()
