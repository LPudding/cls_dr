import timm.models.resnet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dis
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
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, multilabel_confusion_matrix, \
    classification_report

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR, MultiStepLR, \
    StepLR, OneCycleLR
from data_process.process_data import label_list
from models.model import ResNext, SeResNet, EfficientNet, Xception, ResNet, ResNest, Inception_Resnet, Densenet, \
    ResNet_head
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
    IAAAdditiveGaussianNoise, CoarseDropout, Transpose, ColorJitter
)
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter

# from adamp import AdamP
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings

warnings.filterwarnings("ignore")

listt = [
    'onion_gray', 'onion_soft', 'onion_downy', 'onion_rust', 'onion_rphy', 'onion_purple',
    'onion_health', 'ginger_spot', 'ginger_rot', 'ginger_stalk', 'ginger_sheath',
    'ginger_anth', 'ginger_bacrax', 'ginger_health', 'garlic_white', 'garlic_virus', 'garlic_rust', 'garlic_rax',
    'garlic_rphy', 'garlic_purple', 'garlic_health', 'citrus_scab', 'citrus_yellow', 'citrus_shoot', 'citrus_canker',
    'citrus_mela', 'citrus_anth',
    'citrus_health', 'pepper_powdery', 'pepper_virus', 'pepper_brown', 'pepper_gray', 'potato_viral', 'potato_scab',
    'potato_black', 'potato_ring',
    'potato_late', 'potato_early', 'potato_health', 'rice_smut', 'rice_blast', 'rice_disease', 'rice_brown',
    'rice_sheath', 'rice_streak',
    'rice_health', 'banana_scab', 'banana_black', 'banana_top', 'banana_anth', 'banana_wilt', 'banana_leaf',
    'banana_health', 'wheat_powdery', 'wheat_redbil',
    'wheat_takeall', 'wheat_sheath', 'wheat_rust', 'wheat_rax', 'wheat_health', 'pepper_anth', 'pepper_rphy',
    'pepper_health'
]
dic = {}
for idx, i in enumerate(listt):
    dic[i] = idx

# torch.cuda.set_device(1)
gpus = True
device_ids = [0, 1]  # 可用GPU
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')


# 多分类
class TCDataset(Dataset):
    def __init__(self, df, transform=None, mode='train'):
        # self.path = path
        self.df = df
        self.base_dir = self.df['base_dir'].values
        self.source = self.df['source'].values
        self.img_ids = self.df['image_name'].values
        self.labels = self.df['label'].values
        # self.labels = self.df[['MA', 'HE', 'SE', 'EX', 'IRMA', 'NV']].values
        # print(self.labels)
        # self.labels[self.labels > 1] = 1
        # self.img_size = img_size        
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # print(os.path.join(self.path, self.img_ids[index]))
        try:
            # print(os.path.join(self.base_dir[index], self.source[index], self.img_ids[index]))
            # if self.mode == 'train':
            #     img = cv2.imread(os.path.join('Viewers', self.base_dir[index], self.source[index], self.img_ids[index]))
            # # img = cv2.imread(os.path.join(self.base_dir[index], self.source[index], self.img_ids[index]))
            # else:
            # print(os.path.join(self.base_dir[index], self.source[index], self.img_ids[index]))
            img = cv2.imread(os.path.join(self.base_dir[index], self.source[index], self.img_ids[index]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = cv2.imread(os.path.join(self.base_dir[index], self.source[index], self.img_ids[index]))
            # print(os.path.join(
            #     os.path.join('../../../../../dev/shm/w/dr', f'{self.img_ids[index]}.jpg')))  # Original_Images
        #         print(img)
        #         img = get_rgby_image(self.path, self.img_ids[index])
        # x = self._get_image(self.img_ids[index])
        # print(os.path.join(self.path, self.img_ids[index]))
        # print(type(img))
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        #         print(img.shape) # 3,size,size
        label = int(self.labels[index])
        # label = [float(x) for x in self.labels[index]]
        #         label = np.array(y)
        # img = torch.from_numpy(img, dtype=torch.float32)
        label = torch.tensor(label)
        # y = y.sum(axis=0)
        if self.mode == 'test':
            return img
        elif self.mode == 'val':
            return img, label
        else:
            return img, label


def get_score(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return cohen_kappa_score(y_pred, y_true, weights='quadratic')


def get_transforms(args, mode):
    if mode == 'train':
        return Compose([
            RandomResizedCrop(args.image_size, args.image_size, scale=(0.87, 1.15), ratio=(0.7, 1.3), p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(p=0.5, limit=(-180, 180), value=0),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            # ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.2),
            # RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.1), p=0.2),
            # albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
            # albumentations.OpticalDistortion(),
            # AdvancedLineAugmentation(p=0.1),
            # albumentations.OneOf([
            #     albumentations.OpticalDistortion(distort_limit=1.0),
            #     albumentations.ElasticTransform(alpha=3),
            # ], p=0.2),

            Resize(args.image_size, args.image_size),
            # CoarseDropout(p=0.2),
            # albumentations.JpegCompression(quality_lower=80, p=0.5),
            # albumentations.MultiplicativeNoise(p=0.2),
            # A.IAASharpen(p=0.5),
            # Cutout(max_h_size=16, max_w_size=16, num_holes=16, fill_value=(192, 192, 192), p=0.2),
            # Cutout(p=0.2, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
            Normalize(
                mean=[0.426, 0.297, 0.213],
                std=[0.277, 0.202, 0.169],
            ),
            ToTensorV2(),
        ])

    elif mode == 'val' or mode == 'test':
        return Compose([
            Resize(args.image_size, args.image_size),
            # CenterCrop(380,380, p=1.),
            Normalize(
                mean=[0.426, 0.297, 0.213],
                std=[0.277, 0.202, 0.169],
            ),
            ToTensorV2(),
        ])


def classify(predict):
    thresholds = [-0.5 + j for j in range(5)]
    predict = max(predict, thresholds[0])
    for j in reversed(range(len(thresholds))):
        if predict >= thresholds[j]:
            return j


def train_one_epoch(model, train_dataloader, optimizer, loss_fn, scheduler, args):
    # one epoch train
    model.train()
    # scaler = GradScaler()
    train_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    train_losses = 0.0
    # aux_opt = torch.optim.Adam([expt_a, expt_b], lr=0.00002, weight_decay=1e-6, betas=(0.5, 0.999))
    # lr finder
    # all_labels = []
    # all_preds = []
    # losses = []
    # lrs = []
    # optimizer.param_groups[0]['lr'] = 1e-7
    # mult = (1.0 / 1e-7) ** (1/(len(train_dataloader)-1))
    # batch_num = 0
    # avg_loss = 0.0
    for step, (img, label) in train_pbar:
        # print(step)
        # batch_num += 1
        # optimizer.zero_grad()
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        # print(label)
        # print(label.size())
        # torch.amp

        # if epoch < 6:
        #     # mixup opreation
        # img, label_a, label_b, lam = mixup_data(img, label, alpha=0.2)

        # cutmix opreation
        # cutmix_decision = np.random.rand()
        # if cutmix_decision < 0.5:
        #     img, label = cutmix(img, label,alpha=1.0)

        # with autocast():
        out, features = model(img)
        # loss_ce = loss_fn(out, label)  # + loss_fn(out2, label)
        # loss = loss_ce

        # mixup operation
        # loss = mixup_criterion(loss_fn, out, label_a, label_b, lam)
        loss = loss_fn(out, label)
        # cutmix operation
        # if cutmix_decision < 0.5:
        #     loss = loss_fn(out, label[0]) * label[2] + loss_fn(out, label[1]) * (1-label[2])
        # else:
        #     loss = loss_fn(out, label)

        # auc_loss_value = auc_loss(out, label, expt_a, expt_b, alpha)
        # loss += auc_loss_value

        # loss += (0.5 * focal_loss(out, label)) # 组合bce和focal loss
        # 记录loss和对应的lr,发现最好的lr
        # Compute the smoothed loss
        # avg_loss = 0.9 * avg_loss + (1-0.9) *loss.item()
        # smoothed_loss = avg_loss / (1 - 0.9**batch_num)
        # lr = optimizer.param_groups[0]['lr']
        # losses.append(smoothed_loss)
        # lrs.append(lr)

        train_losses += loss.item()
        # print(args.accumulation_steps)
        # exit(0)
        # if args.accumulation_steps > 1:
        #     loss = loss / args.accumulation_steps

        # scaler.scale(loss).backward()
        loss.backward()
        # 梯度裁剪前需要调用下面这个
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
        # all_preds.extend([out.sigmoid().detach().to('cpu').numpy()]) # 对记录的模型输出进行sigmoid处理
        # all_labels.extend([label.detach().cpu('cpu').numpy()])
        # optimizer.zero_grad()
        # loss.backward()
        # pytorch 自带加速
        # scaler.scale(loss).backward()
        if (step + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # if alpha.grad is not None:
            #     # alpha.data = alpha.data + 0.00002*alpha.grad.data
            #     alpha.data = torch.relu(alpha.data + 0.00002*alpha.grad.data)
            #     alpha.grad.data *= 0 

            # optimizer.zero_grad()
            # aux_opt.zero_grad()
        # OneCycle scheduler
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # 测试lr和loss的关系选择lr
        # lr = lr * mult
        # optimizer.param_groups[0]['lr'] = lr
        # apex加速训练
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        # optimizer.step()
    # all_labels = np.concatenate(all_labels)
    # all_preds = np.concatenate(all_preds)
    # train_acc = np.mean(all_labels==all_preds)
    # train_score = get_score(all_labels, all_preds)
    # plt.plot(lrs, losses)
    # plt.xscale('log')
    # plt.savefig('./152d_loss_lr.jpg')
    # 清空GPU缓存
    torch.cuda.empty_cache()

    return train_losses / len(train_dataloader)


def val_one_epoch(model, val_dataloader, loss_fn):
    # one epoch val
    # print('testing!!!')
    model.eval()
    val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    val_losses = 0.0
    all_preds_softmax = []
    all_preds = []
    all_labels = []
    precise_avg = Mean_calc()
    recall_avg = Mean_calc()
    with torch.no_grad():
        for step, (img, label) in val_pbar:
            bs = img.shape[0]
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            out, _ = model(img)
            loss = loss_fn(out, label)

            val_losses += loss.item()
            # 保存预测的标签和样本label
            # print(out)
            # all_preds.extend([np.array([classify(p.item()) for p in out])])
            # print(all_preds)
            # exit(0)
            all_preds.extend([out.to('cpu').numpy()])  # 对记录的模型输出进行softmax处理
            all_preds_softmax.extend([torch.argmax(out, dim=1).to('cpu').numpy()])  # 对记录的模型输出进行softmax处理
            # preds = out.sigmoid().to('cpu').numpy()
            # preds[preds >= 0.5] = 1
            # preds[preds < 0.5] = 0
            # print(preds)
            # exit(0)
            # acc = np.mean(np.equal(preds, label.to('cpu').numpy()))
            # precise_avg.update(acc * bs, bs)
            # all_preds.extend([preds])
            all_labels.extend([label.long().to('cpu').numpy()])

            # for pred, gt in zip(torch.max(out.sigmoid(), 1)[1], label):
            #     pred = pred.item()
            #     gt = gt.item()
            #     if pred == 1:
            #         if pred == gt:
            #             precise_avg.update(1)
            #             recall_avg.update(1)
            #         else:
            #             precise_avg.update(0)
            #     if gt == 1 and pred == 0:
            #         recall_avg.update(0)
        all_preds_softmax = np.concatenate(all_preds_softmax)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # print(all_preds_softmax.shape)
        # print(all_labels.shape)
        # print(all_preds)
        # print(all_labels)
        # print(all_labels.shape)
        # val_acc = np.mean(np.argmax(all_preds, axis=1) == all_labels)
        # precise = precise_avg.get_mean() * 100
        # recall = recall_avg.get_mean() * 100
        # f1 = 2 * precise * recall / (precise + recall)
        # print("precise:%.2f %% \t recall:%.2f %%\t f1:%.2f %% patch_num:%d" % (precise, recall, f1, precise_avg.count))
        # print('confusion_matrix:')
        # print("multilabel_cm: MA HE SE EX IRMA NV")
        # print(multilabel_confusion_matrix(all_labels, all_preds))
        # print("val_acc:", val_acc)
        print(classification_report(all_labels, all_preds_softmax))
        val_score = get_score(all_labels, all_preds)
        # val_score = accuracy_score(all_labels, all_preds)
        # val_score = precise_avg.get_mean()
        # val_score=100
    # return val_losses / len(val_dataloader), precise / 100, all_preds

    return val_losses / len(val_dataloader), val_score, all_preds


def train_loop(model, train_loader, val_loader, val_df, train_loss_fn, val_loss_fn, optimizer, scheduler, fold, args, start_epoch=0, best_score=0.0):
    # main train function
    best_val_score = best_score
    best_val_loss = np.inf
    early_stop_count = 0
    # scaler = GradScaler()

    # Define expt_a, expt_b and alpha
    # expt_a = torch.zeros(11, dtype=torch.float32, device=device, requires_grad=True)
    # expt_b = torch.zeros(11, dtype=torch.float32, device=device, requires_grad=True)
    # alpha = torch.zeros(11, dtype=torch.float32, device=device)+0.1
    # alpha.requires_grad = True

    for epoch in range(args.epochs):
        # 另外一种cosin调整方法
        # optimizer.param_groups[0]['lr'] = adjust_learning_rate(5e-4, 0, args.epochs-1, optimizer, epoch)
        # if epoch == 10:
        #     optimizer.param_groups[0]['lr'] = 5e-5
        if epoch == 0 and start_epoch != 0:
            epoch = start_epoch
        train_epoch_loss = train_one_epoch(model, train_loader, optimizer, train_loss_fn, scheduler, args)

        logger.info('Epoch {}: Train loss is {:.4f}'.format(epoch + 1, train_epoch_loss))

        val_epoch_loss, val_epoch_score, val_epoch_preds = val_one_epoch(model, val_loader, val_loss_fn)

        logger.info('Epoch {}: Val loss is {:.4f}'.format(epoch + 1, val_epoch_loss))

        logger.info('Epoch {}: Val score is {:.4f}'.format(epoch + 1, val_epoch_score))

        logger.info('Epoch {}: Learning Rate is {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        tag_dict = {'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss, 'val_score': val_epoch_score,
                    'lr': optimizer.param_groups[0]['lr']}
        writer.add_scalars('record', tag_dict, epoch)
        # exit(0)

        # if epoch < 9:
        # scheduler step
        if scheduler != None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_epoch_score)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            # elif isinstance(scheduler, GradualWarmupSchedulerV2):
            #     scheduler.step()
            #     if epoch == 1:
            #         scheduler.step()
            elif isinstance(scheduler, MultiStepLR):
                scheduler.step()

        # elif epoch == 10:
        #     optimizer.param_groups[0]['lr']=5e-5
        #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=5e-6, last_epoch=-1)
        #     print('now!')
        # elif epoch < 13:
        #     optimizer.param_groups[0]['lr']=5e-5
        # elif epoch < 13:
        #     optimizer.param_groups[0]['lr'] = 3e-5
        # elif epoch < 15:
        #     optimizer.param_groups[0]['lr'] = 1e-5
        # elif epoch < 16:
        #     optimizer.param_groups[0]['lr'] = 5e-6
        # else:
        #     optimizer.param_groups[0]['lr'] = 2e-6

        #         if val_epoch_loss < best_val_loss:
        #             best_val_loss = val_epoch_loss
        #         if val_epoch_score > best_val_score:
        #             best_val_score = val_epoch_score
        #             logger.info('Best score is {:.4f}'.format(best_val_score))
        #             logger.info('Update model in epoch {}'.format(epoch+1))
        # output_dir = os.path.join(args.saved_dir, args.model_name)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # saved_model_path = os.path.join(output_dir, 'fold_{}_best.pth'.format(fold))
        #
        # #         model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        # if gpus:
        #     checkpoint = {
        #         'epoch': epoch + 1,
        #         'state_dict': model.module.state_dict(),
        #         # 'amp': amp.state_dict()
        #     }
        # else:
        #     checkpoint = {
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         # 'amp': amp.state_dict()
        #     }
        # torch.save(checkpoint, saved_model_path)
        #         early_stop_count = 0

        #         # Early stopping operation
        #         if val_epoch_score < best_val_score:
        #             early_stop_count += 1
        #             if early_stop_count == args.early_stop_patience:
        #                 logger.info('Fold {} training stop in epoch {}'.format(fold, epoch+1))
        #                 break

        if val_epoch_score > best_val_score:
            best_val_score = val_epoch_score
            logger.info('Best score is {:.4f}'.format(best_val_score))
            logger.info('Update model in epoch {}'.format(epoch + 1))
            output_dir = os.path.join(args.saved_dir, args.model_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # saved_model_path = os.path.join(output_dir, 'fold_{}_best.pth'.format(fold))
            saved_model_path = os.path.join(output_dir, 'epoch_{}_best.pth'.format(epoch))
            model_save = (model.module if hasattr(model, "module") else model)  # 判断是否是多卡训练
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model_save.state_dict(),
                'preds': val_epoch_preds,
                # 'amp': amp.state_dict()                    
            }
            torch.save(checkpoint, saved_model_path)
            early_stop_count = 0
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
        # Early stopping operation
        if val_epoch_loss > best_val_loss:
            early_stop_count += 1
            if early_stop_count > args.early_stop_patience:
                logger.info('Fold {} training stop in epoch {}'.format(fold, epoch + 1))
                break

        # 清空GPU缓存
        torch.cuda.empty_cache()


def get_csv(data_dir):
    classes = os.listdir(data_dir)
    #     dic = {}
    #     for idx, i in enumerate(classes):
    #         dic[i] = idx
    images = []
    labels = []
    for cls in classes:
        imgs = os.listdir(os.path.join(data_dir, cls))
        #         print(imgs)
        for i in imgs:
            try:
                img = cv2.imread(os.path.join(data_dir, cls, i))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(os.path.join(data_dir, cls, i))
                labels.append(dic[cls])
            except:
                print(os.path.join(os.path.join(data_dir, cls, i)))
    df = pd.DataFrame({"images": images, "labels": labels})
    df.to_csv('train.csv', index=False)


def main(args):
    # main function
    seed_everything()

    train_df_data = pd.read_csv('trainLabels15.csv')
    df_data = pd.read_csv('testLabels15.csv')
    val_df_data = df_data[df_data.Usage == 'Public'].reset_index(drop=True)
    test_df_data = df_data[df_data.Usage == 'Private'].reset_index(drop=True)


    if args.debug:
        train_df_data = train_df_data.sample(frac=0.02)
        val_df_data = val_df_data.sample(frac=0.02)
        test_df_data = test_df_data.sample(frac=0.02)
        
    train_transforms = get_transforms(args, mode='train')
    val_transforms = get_transforms(args, mode='val')
    
    oof_df = pd.DataFrame()  # 用于存储每个fold的,每个最佳模型的val结果

    fold = args.fold

    logger.info('------------Fold {} Training is start!-------------'.format(fold))
    # 加载数据

    print(len(test_df_data))
    
    # 这里记得更改,目前是用训练好的模型再次训练没有增强的模型
    train_dataset = TCDataset(train_df_data, transform=train_transforms, mode='train')
    val_dataset = TCDataset(val_df_data, transform=val_transforms, mode='val')
    test_dataset = TCDataset(test_df_data, transform=val_transforms, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    # 设置模型及优化器
    # 保证模型每次初始化一致
    seed_everything()
    logger.info('The model is using {}'.format(args.model_name))
    
    model = ResNet(model_name=args.model, pretrained=True, n_class=5)

    logger.info('model\' architecture: ')
    logger.info(model)

    if gpus:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # kaggle kernel method
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = Lookahead(RAdam(model.parameters(),lr=args.lr), alpha=0.5, k=5)
    # optimizer = AdaBelief(model.parameters(), lr=1e-3, weight_decay=5e-4) # 待测试
    # optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # aux_opt = torch.optim.Adam([expt_a, expt_b], lr=0.00002, weight_decay=1e-5, betas=(0.5, 0.999))

    # scheduler进行lr调整
    #     scheduler = None
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5, min_lr=1e-6, verbose=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=2e-4, last_epoch=-1)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=6, eta_min=1e-6, last_epoch=-1) #目前resnext,t_max=4最好
    # scheduler = MultiStepLR(optimizer, milestones=[11,14,18], gamma=0.1) #目前最高，但是待考量，发现不同模型分数相似
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    # scheduler = OneCycleLR(optimizer, max_lr=5e-4, epochs=30, steps_per_epoch=len(train_loader))
    # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler)

    # loss function
    val_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = LabelSmoothing()
    # train_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.387, 1]))
    # train_loss_fn = nn.CrossEntropyLoss()
    # train_loss_fn = SmoothingBCELossWithLogits(smoothing=0.0001)
    #     train_loss_fn = FocalLoss()
    # train_loss_fn = nn.MSELoss()
    # loss_fn = LovaszLoss()
    # phat = torch.true_divide(torch.sum(torch.from_numpy(train_dataset.labels), dim=0), train_dataset.labels.shape[0])
    # auc_loss = DeepAUC(phat.to(device))

    # 使用apex加速训练
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # 记录参数以及优化器相关
    logger.info('The optimizer is {}'.format(optimizer))
    logger.info('The scheduler is {}'.format(scheduler))
    logger.info('The Train loss function is {}'.format(train_loss_fn))
    logger.info('The Val loss function is {}'.format(val_loss_fn))
    # val_epoch_loss, val_epoch_score, val_epoch_preds = val_one_epoch(model, test_loader, val_loss_fn)
    # logger.info('Epoch {}: Val score is {:.4f}'.format(37, val_epoch_score))
    # exit(0)

    # 训练和保存结果
    #     val_loader, val_df, val_loss_fn = None, None, None
    val_df = None
    train_loop(model, train_loader, val_loader, val_df, train_loss_fn, val_loss_fn, optimizer, scheduler, fold, args)


if __name__ == '__main__':
    args = get_argparse()
    args.debug = False
    args.model_name = 'r50_cls_25e_32bs_labelsmooth'
    args.saved_dir = f'./models'
    args.model = 'resnet50'
    args.fold = 0
    args.logs_dir = f'./logs/{args.model_name}'
    args.epochs = 25
    args.batch_size = 32
    args.lr = 2e-3
    args.weight_decay = 5e-4
    args.momentum = 0.9
    args.early_stop_patience = 10
    args.image_size = 512
    args.num_workers = 8
    logger = get_logger(args.logs_dir)
    writer = SummaryWriter(args.logs_dir)
    logger.info(json.dumps(vars(args), indent=4, ensure_ascii=False, sort_keys=False))
    main(args)
