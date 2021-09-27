# %%
import torch
from pathlib import Path
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from net.FastRCNN import FastSCNN
from data.cityscapes import CityScapesSeg
from data import img_transformer as T
from config import Config
from torch.optim import lr_scheduler as lrs
import torch.utils.data as data
from metrics.confusion_matrix import ConfusionMatrix
from metrics.IOU import IOU
from tqdm import tqdm
import numpy as np
import os
import re


train_tf = T.ComposedTransform([
    T.RandomFlip(Config['probality_of_flip']),
    T.GaussianBlur(Config['probality_of_gaussian_blur']),
    T.PadResize(Config['pad_resize']),
    T.AllToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_tf = T.ComposedTransform([
    T.PadResize(Config['pad_resize']),
    T.AllToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_ds = CityScapesSeg(Config['data_root_path'], transform=train_tf)
train_dl = data.DataLoader(train_ds, Config['batch_size'],
                           Config['shuffle'], pin_memory=True, num_workers=0)

val_ds = CityScapesSeg(Config['data_root_path'],
                       split='val',  transform=val_tf)
val_dl = data.DataLoader(
    val_ds, Config['batch_size_of_val'], Config['shuffle'], pin_memory=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FastSCNN(train_ds.class_num)

criterion = nn.CrossEntropyLoss(ignore_index=255)

CM = ConfusionMatrix()
IOU = IOU()

optimizer = torch.optim.SGD(model.parameters(),
                            lr=Config['lr'],
                            momentum=Config['momentum'],
                            weight_decay=Config['weight_decay'])

# lr scheduling
lr_scheduler = lrs.StepLR(optimizer, step_size=7, gamma=0.1)


start_epoch = -1
losses = []
val_losses = []
ious = []
mious = []
if Config['is_resume']:
    checkpoint_dir = Path('./checkpoint/train_data')
    paths = [path.stem if path.is_file else None for path in checkpoint_dir.iterdir()]
    if len(paths) != 0:  # 存在保存点
        num = [int(re.findall(r'\d+', path)[0]) for path in paths]
        num = max(num)
        checkpoint = torch.load(
            checkpoint_dir / Path('ckpt_epoch_{}'.format(num)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        losses = checkpoint['losses']
        val_losses =  checkpoint['val_losses']
        ious = checkpoint['ious']
        mious = checkpoint['mious']

for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

model = model.to(device)
# %%
for epoch in tqdm(range(start_epoch+1, Config['epochs']), desc='epochs: '):
    # 对于每个batch
    for idx, (imgs, masks) in tqdm(enumerate(train_dl), desc='train: '):
        # 获得输出
        imgs = imgs.to(device)
        masks = masks.to(device)
        pres = model(imgs)[0]  # tuple(res)
        loss = criterion(pres, masks)
        # 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # 输出loss
        tqdm.write("loss: {}".format(loss))
        losses.append(loss.detach().cpu().numpy())
    del imgs, masks

    # 测试
    if epoch % Config["interval_of_val"] == 0:
        cm_all = []
        for idx, (imgs, masks) in tqdm(enumerate(val_dl), desc='val: '):
            imgs = imgs.to(device)
            masks = masks.to(device)
            pres = model(imgs)[0]  # tuple(res)
            loss = criterion(pres, masks)
            tqdm.write("val loss: {}".format(loss))
            val_losses.append(loss.detach().cpu().numpy())

            pres = torch.argmax(pres.detach().cpu(), dim=1,
                                keepdim=False).flatten().numpy()
            masks = masks.flatten().cpu().detach().numpy()
            cm, labels = CM.get_confusion_matrix(
                pres, masks, labels=list(range(19))+[255])
            cm_all.append(cm)
            labels_all = labels
        cm_all = sum(cm_all)
        trainID_labels = {
            int(k): v for k, v in Config["trainID_labels"].items()}

        val_data_dir = Path('./checkpoint/val_dir')
        if not os.path.isdir(val_data_dir):
            os.mkdir(val_data_dir)
        CM.save_fig(val_data_dir / Path('confusion_matrix_epoch_{}'.format(epoch)),
                    cm_all, labels_all, labels=trainID_labels)
        iou, miou = IOU.generate_iou(cm_all)
        ious.append(iou)
        mious.append(miou)

    # 收集最好的
    mious_ndarray = np.array(mious)
    if mious_ndarray.size != 0:
        if int(mious_ndarray.argmax()) == mious_ndarray.size - 1:  # 最新的一次是最好的一次
            checkpoint = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "losses": losses,
                "val_losses": val_losses,
                "ious": ious,
                "mious": mious
            }
            train_data_dir = Path('./checkpoint/train_data')
            if not os.path.exists(train_data_dir):
                os.mkdir(train_data_dir)
            torch.save(checkpoint, train_data_dir /
                       Path('best_ckpt_epoch'.format(epoch)))

    if epoch % Config["interval_of_save_weight"] == 0:
        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "losses": losses,
            "val_losses": val_losses,
            "ious": ious,
            "mious": mious
        }
        train_data_dir = Path('./checkpoint/train_data')
        if not os.path.exists(train_data_dir):
            os.mkdir(train_data_dir)
        torch.save(checkpoint, train_data_dir /
                   Path('ckpt_epoch_{}'.format(epoch)))
