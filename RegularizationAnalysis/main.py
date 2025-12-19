from easydict import EasyDict as edict
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import IdentificationNet
import numpy as np
import torch
from callback import StepLossAccInfoTorch
import torch.nn as nn
import matplotlib.pyplot as plt

def train(batchnorm=False, dropout=False, l2=False, earlystop=False):
    cfg.count = 0
    cfg.stop = False
    cfg.min_val_loss = float('inf')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MyDataset(cfg, 'train')
    test_dataset = MyDataset(cfg, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False
    )

    model = IdentificationNet(
        use_bn=batchnorm,
        use_dropout=dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    if l2:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr
        )

    monitor = StepLossAccInfoTorch(
        model=model,
        eval_loader=test_loader,
        print_iter=125,
        early_stop=earlystop,
        cfg=cfg,
    )

    print("============== Starting Training ==============")

    global_step = 0
    num_epochs = 30

    for epoch in range(num_epochs):
        for x, y in train_loader:
            global_step += 1

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            monitor.step_end(
                cur_epoch=epoch,
                cur_step=global_step,
                loss_value=loss.item()
            )

            if cfg.stop:
                break

        if cfg.stop:
            break

    final_acc = monitor.evaluate()
    metric = {'acc': final_acc}
    print(metric)

    return (
        np.array(monitor.loss),
        np.array(monitor.acc),
        metric
    )

def drawCurves(loss1, label1, loss2, label2, loss3, label3):
    plt.figure(figsize=(20,10))
    
    x1 = np.linspace(1, len(loss1), len(loss1))
    plt.plot(x1, loss1, linewidth=1, label=label1)
    
    x2 = np.linspace(1, len(loss2), len(loss2))
    plt.plot(x2, loss2, linewidth=1, label=label2)
    
    x3 = np.linspace(1, len(loss3), len(loss3))
    plt.plot(x3, loss3, linewidth=1, label=label3)
    
    # my_y_ticks = np.arange(0, 1, 0.1)
    # plt.yticks(my_y_ticks)
    plt.xlabel("iter") #xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.ylabel('loss')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    cfg = edict({
        'train_size': 60000,# 训练集大小
        'test_size': 10000,# 测试集大小
        'channel': 1,# 图片通道数
        'image_height': 28,# 图片高度
        'image_width': 28,# 图片宽度
        'batch_size': 32,
        'num_classes': 10,# 分类类别
        'lr': 0.001,# 学习率
        'epoch_size': 20,# 训练次数
        'data_dir_train': os.path.join('fashion-mnist', 'train'),
        'data_dir_test': os.path.join('fashion-mnist', 'test'),
        'count': 0,
        'min_val_loss': 1,
        'stop': False,
        'MAX_COUNT': 10,
        'weight_decay': 0.00001 # L2正则化系数
    })

    fc_loss, fc_acc, _ = train(batchnorm=False, dropout=False, l2=False, earlystop=False)
    dropout_loss, dropout_acc, _ = train(batchnorm=True, dropout=True, l2=False, earlystop=False)
    l2_loss, l2_acc, _ = train(batchnorm=True, dropout=False, l2=True, earlystop=False)
    drawCurves(fc_loss, 'fc_loss', dropout_loss, 'dropout_loss', l2_loss, 'l2_loss')
    drawCurves(fc_acc, 'fc_acc', dropout_acc, 'dropout_acc', l2_acc, 'l2_acc')
