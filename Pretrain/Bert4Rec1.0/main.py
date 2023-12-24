import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    # 调用utils中的setup_train函数
    # args来自options文件
    # 生成保存实验的路径名
    export_root = setup_train(args)
    # 调用dataloaders文件夹中__init__.py文件里的dataloader_factory函数
    train_loader, val_loader, test_loader = dataloader_factory(args)
    # 调用models文件夹中__init__.py文件里的model_factory函数
    # 还没有调用model的forward函数
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    # 训练 调用train函数
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    # 测试
    if test_model:
        # 测试 调用test函数
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
