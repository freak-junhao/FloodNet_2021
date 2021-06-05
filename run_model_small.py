from models.load_data import VqaDataset
from models.mcan_baseline import VqaNet
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
    token_size = dataset.token_size
    pre_emb = dataset.pre_emb
    ans_size = dataset.ans_size

    # init model
    model = VqaNet(config, pre_emb, token_size, ans_size)
    print(model)
    model.cuda()
    # device = torch.device("cuda:1")
    # model.to(device)
    model.train()

    model = nn.DataParallel(model, device_ids=config['DEVICES'])
    loss_fn = nn.BCELoss(reduction='sum').cuda()

    if ('ckpt_' + config['version']) in os.listdir(config['CKPTS_PATH']):
        shutil.rmtree(config['CKPTS_PATH'] + 'ckpt_' + config['version'])

    os.mkdir(config['CKPTS_PATH'] + 'ckpt_' + config['version'])

    optim = get_optim(config, model, data_size)
    start_epoch = 0

    loss_sum = 0
    named_params = list(model.named_parameters())
    grad_norm = np.zeros(len(named_params))

    dataloader = DataLoader(
        dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEM'],
        drop_last=True
    )

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
        for step, (image_data, ques_ix, ans_ix) in enumerate(dataloader):

            optim.zero_grad()

            image_data = image_data.cuda()
            ques_ix = ques_ix.cuda()
            ans_ix = ans_ix.cuda()

            pred = model(image_data, ques_ix)
            loss = loss_fn(pred, ans_ix)
            loss.backward()

            loss_sum += loss.cpu().data.numpy()

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
        if epoch_finish % 10 == 0:
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

        # Logging
        logfile = open(
            config['LOG_PATH'] +
            'log_run_' + config['version'] + '.txt',
            'a+'
        )
        logfile.write(
            'epoch = ' + str(epoch_finish) +
            '  loss = ' + str(loss_sum / data_size) +
            '\n' +
            'lr = ' + str(optim._rate) +
            '\n\n'
        )
        logfile.close()

        # Eval after every epoch
        # if dataset_eval is not None:
        #     self.eval(
        #         dataset_eval,
        #         state_dict=net.state_dict(),
        #         valid=True
        #     )

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))


def val(config, dataset):
    # Load parameters
    ckpt_path = config['CKPTS_PATH'] + 'ckpt_' + config['version'] + \
           '/epoch' + str(config['CKPT_EPOCH']) + '.pkl'
    print('Loading ckpt {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)['state_dict']
    print('Finish!')

    pred_list = []
    data_size = dataset.data_size
    token_size = dataset.token_size
    pre_emb = dataset.pre_emb
    ans_size = dataset.ans_size

    model = VqaNet(config, pre_emb, token_size, ans_size)
    model.cuda()
    model.eval()
    model = nn.DataParallel(model, device_ids=config['DEVICES'])

    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(state_dict)

    dataloader = DataLoader(
        dataset,
        batch_size=config['EVAL_BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )

    for step, (image_data, ques_ix, ans) in enumerate(dataloader):
        print("\rEvaluation: [step %4d/%4d]" % (
            step,
            int(data_size / config['EVAL_BATCH_SIZE']),
        ), end='          ')

        image_data = image_data.cuda()
        ques_ix = ques_ix.cuda()

        pred = model(image_data, ques_ix)
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)

        for ans_ix in pred_argmax:
            pred_list.append(ans_ix)

    print('Predict Finished! ')

    ans_list = [dataset.ans_ix.get(str(pred)) + '\n' for pred in pred_list]

    with open('results/answers/_{}.txt'.format(config['version']), 'w') as f:
        for i in ans_list:
            f.write(i)


def run_model():
    args = parse_args()
    config = load_configs(args.cfg_path)
    config['version'] = args.version
    config['run_mode'] = args.run_mode

    if args.run_mode == 'train':
        config['image_path'] += 'Train_Image/'
        config['question_path'] += 'Training Question.json'

        print('Loading datasets -->')
        dataset = VqaDataset(config)

        train(config, dataset)

    else:
        config['image_path'] += 'Test_Image/'
        config['question_path'] += 'Test_Question.json'

        print('Loading datasets -->')
        dataset = VqaDataset(config)

        val(config, dataset)


if __name__ == '__main__':
    run_model()
