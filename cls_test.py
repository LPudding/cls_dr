import timm.models.resnet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
# from apex import amp
from tqdm import tqdm
import numpy as np
import math
import cv2
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from sklearn.metrics import cohen_kappa_score
import json
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, multilabel_confusion_matrix

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR, MultiStepLR, \
    StepLR, OneCycleLR
from data_process.process_data import label_list
from models.model import ResNext, SeResNet, EfficientNet, Xception, ResNet, ResNest, Inception_Resnet, Densenet, \
    ResNet_head
from models.wsol.resnet import ResNet18Adl
from utils.utils import seed_everything, get_logger, get_result, adjust_learning_rate, mixup_criterion, mixup_data, \
    cutmix, Mean_calc
from utils.get_args import get_argparse
# from schedulers.sched import GradualWarmupSchedulerV2
from optimizers.radam import RAdam
from optimizers.lookahead import Lookahead
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
# from adabelief_pytorch import AdaBelief
import matplotlib.pyplot as plt
from losses.loss import FocalLoss, LovaszLoss, SmoothingBCELossWithLogits, DeepAUC, info_nce_loss, LabelSmoothing
import albumentations
import albumentations as A
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, CenterCrop,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, HueSaturationValue,
    IAAAdditiveGaussianNoise, CoarseDropout, Transpose
)
from albumentations.pytorch import ToTensorV2
from cls_train import get_transforms, TCDataset
import torchvision.transforms as transforms
# from adamp import AdamP
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings

gpus = True
device_ids = [0, 1]  # 可用GPU
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')


def get_score(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return cohen_kappa_score(y_pred, y_true, weights='quadratic')


def eval(model, val_dataloader):
    model.eval()
    val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    all_preds_softmax = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for step, (img, label) in val_pbar:
            bs = img.shape[0]
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            out, _ = model(img)
            all_preds.extend([out.to('cpu').numpy()])  # 对记录的模型输出进行softmax处理
            all_preds_softmax.extend([torch.softmax(out, dim=1).to('cpu').numpy()])  # 对记录的模型输出进行softmax处理
            all_labels.extend([label.long().to('cpu').numpy()])
            
        all_preds_softmax = np.concatenate(all_preds_softmax)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_score = get_score(all_labels, all_preds)

    return val_score, all_preds_softmax, all_labels



def main(args):
    # main function
    seed_everything()
    # df_data = pd.read_csv(os.path.join('fold-5-attr.csv'))
    # df_data = df_data[df_data.fold == 0]
    # df_data = df_data[df_data.ill > 1]
    data_path = args.data_dir
    oof_df = pd.DataFrame()  # 用于存储每个fold的,每个最佳模型的val结果

    fold = args.fold

    # 加载数据
    train_df_data = pd.read_csv('trainLabels15.csv')
    df_data = pd.read_csv('testLabels15.csv')
    val_df_data = df_data[df_data.Usage == 'Public'].reset_index(drop=True)
    test_df_data = df_data[df_data.Usage == 'Private'].reset_index(drop=True)

    train_transforms = get_transforms(args, mode='train')
    val_transforms = get_transforms(args, mode='val')
    
    train_dataset = TCDataset(train_df_data, transform=val_transforms, mode='val')
    val_dataset = TCDataset(val_df_data, transform=val_transforms, mode='val')
    test_dataset = TCDataset(test_df_data, transform=val_transforms, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)
    # print(val_df_data['labels'].value_counts())
    print(len(test_df_data))
    #     print('val_loader', len(val_loader))
    # 设置模型及优化器
    # 保证模型每次初始化一致
    seed_everything()
    model = ResNet(model_name=args.model, pretrained=True, n_class=5)
    model.load_state_dict(torch.load(os.path.join('models', 'r50_cls_25e_32bs_labelsmooth',
                                                  'epoch_24_best.pth'))['state_dict'])
    model.to(device)
    model.eval()
    
    # if args.post:
        # train_score, train_preds, train_labels = eval(model, train_loader)
        # print('train qwk: {}'.format(train_score))
        
        # val_score, val_preds, val_labels = eval(model, val_loader)
        # print('val qwk: {}'.format(val_score))
    
    test_score, test_preds, test_labels = eval(model, test_loader)
    print('test qwk: {}'.format(test_score))
    
    if args.post:
        # train
        # train = np.concatenate([train_df_data['image_name'].values.reshape(-1, 1), train_preds, train_labels.reshape(-1, 1)], axis=1)
        # train_post = pd.DataFrame(train, columns=['image', 'pre0', 'pre1', 'pre2', 'pre3', 'pre4', 'level'])
        # train_post['image'] = train_post['image'].apply(lambda x: x.split('.')[0])
        # train_post.to_csv('train_post.csv', index=False)
        
        # test
        # val = np.concatenate([val_df_data['image_name'].values.reshape(-1, 1), val_preds, val_labels.reshape(-1, 1)], axis=1)
        test = np.concatenate([test_df_data['image_name'].values.reshape(-1, 1), test_preds, test_labels.reshape(-1, 1)], axis=1)
        test_sum = np.concatenate([test], axis=0)
        test_post = pd.DataFrame(test_sum, columns=['image', 'pre0', 'pre1', 'pre2', 'pre3', 'pre4', 'level'])
        test_post['image'] = test_post['image'].apply(lambda x: x.split('.')[0])
        test_post.to_csv('test_post.csv', index=False)
        
        
        
    
    
    
    


if __name__ == '__main__':
    args = get_argparse()
    args.debug = False
    # args.data_dir = '../../../DataSets/'
    # args.data_dir = 'datasets/origin_image/'
    args.data_dir = 'Viewers'

    args.model_name = 'r50_patch_cls_80e_32bs_labelsmooth'
    args.saved_dir = f'./models'
    args.model = 'resnet50'
    args.fold = 0
    args.image_size = 512
    args.batch_size = 16
    args.post = True
    # focal_loss = FocalLoss()
    # auc_loss = DeepAUC()
    main(args)
