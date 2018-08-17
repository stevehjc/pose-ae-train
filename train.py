import sys
import time
import os
import tqdm
from os.path import dirname

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import pickle
import torch
import importlib
import argparse

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='pose', help='task to be trained')
    # 调试时使用,调试时需要设置命令行参数的默认值，以默认值进入调试
    parser.add_argument('-c', '--continue_exp', type=str, default='pretrained',help='continue exp') 
    # parser.add_argument('-c', '--continue_exp', type=str,help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--mode', type=str, default='single', help='scale mode')
    args = parser.parse_args()
    return args

def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pth.tar')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save(config):
    resume = os.path.join('exp', config['opt'].exp)
    resume_file = os.path.join(resume, 'checkpoint.pth.tar')

    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print('=> save checkpoint')

def train(train_func, data_func, config, post_epoch=None):
    while True:
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        for phase in ['train', 'valid']:
            num_step = config['train']['{}_iters'.format(phase)]
            generator = data_func(phase)
            print('start', phase, config['opt'].exp)

            show_range = range(num_step)
            show_range = tqdm.tqdm(show_range, total = num_step, ascii=True)
            batch_id = num_step * config['train']['epoch']
            for i in show_range:
                datas = next(generator)
                outs = train_func(batch_id + i, config, phase, **datas)
        config['train']['epoch'] += 1
        save(config)

def init():
    """
    import configurations specified by opt.task

    task.__config__ contains the variables that control the training and testing
    make_network builds a function which can do forward and backward propagation

    please check task/base.py
    """
    opt = parse_command_line()
    # import_module 类似与Python关键字引入模型：import **。优点是带参数，可以在运行过程中引入某一模型
    task = importlib.import_module('task.' + opt.task) #载入模型配置文件 task/pose.py
    exp_path = os.path.join('exp', opt.exp) #实验输出文件

    config = task.__config__
    # 如果输出文件不存在，尝试递归创新文件夹
    try: os.makedirs(exp_path)
    except FileExistsError: pass

    config['opt'] = opt  # 给config增加一项'opt'
    config['data_provider'] = importlib.import_module(config['data_provider']) #引入data_provider对应的coco数据处理 data/coco_pose/dp.py

    func = task.make_network(config)
    reload(config)
    return func, config

def main():
    func, config = init()
    data_func = config['data_provider'].init(config)
    train(func, data_func, config)

if __name__ == '__main__':
    main()
