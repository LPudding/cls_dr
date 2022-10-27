from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
import random
import os
from numpy.lib.function_base import average
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from data_process.process_data import label_list
import torch.distributed as dist
import logging
import math


class Mean_calc:
    def __init__(self):
        self.count = 0
        self.sum = 0

    def update(self, value, num=1):
        self.sum += value
        self.count += num

    def get_mean(self):
        return self.sum / self.count


def get_logger(logs_dir):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s -  %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    log_file_path = os.path.join(logs_dir, 'train.log')
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(log_format)
    handler2 = FileHandler(filename=log_file_path)
    handler2.setFormatter(log_format)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True


# rank average,可能有些地方不对,结果提交之后很不好
# def rank_average(data):
#     # data是预测好的numpy数组
# res = data.argsort(axis=0).argsort(axis=0)
# return res


#     return res

def get_score(y_true, y_pred):
    # scores = []
    # for i in range(y_true.shape[1]):
    #     score = average_precision_score()(y_true[:,i], y_pred[:,i])
    #     scores.append(score)
    # avg_score = np.mean(scores)
    #     y_pred = np.argmax(y_pred, axis=1)
    #     return accuracy_score(y_true, y_pred)

    return average_precision_score(y_true, y_pred)


# def get_score(labels, preds):
#     return np.mean([roc_auc_score(labels[:, i], preds[:, i]) for i in range(11)])

def get_result(result_df):
    # 计算local CV
    preds = result_df[['pred_{}'.format(c) for c in label_list]].values
    labels = result_df[label_list].values
    score = get_score(labels, preds)
    return score


def reduce_mean(tensor, nprocs):
    # 多卡训练时的loss和acc等评价指标转换
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def adjust_learning_rate(init_lr, start, end, optimizer, epoch):
    # 训练过程中在某个过程更改成另外一个cosin scheduler
    lr = optimizer.param_groups[0]['lr']
    if start <= epoch < end:
        lr = init_lr * (math.cos((epoch - start) / (end - start) * math.pi) + 1) / 2
    return lr


# mixup opreation
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets


# def find_lr(init_value = 1e-8, final_value=10., beta = 0.98):
#     num = len(trn_loader)-1
#     mult = (final_value / init_value) ** (1/num)
#     lr = init_value
#     optimizer.param_groups[0]['lr'] = lr
#     avg_loss = 0.
#     best_loss = 0.
#     batch_num = 0
#     losses = []
#     log_lrs = []
#     for data in trn_loader:
#         batch_num += 1
#         #As before, get the loss for this mini-batch of inputs/outputs
#         inputs,labels = data
#         inputs, labels = Variable(inputs), Variable(labels)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         #Compute the smoothed loss
#         avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
#         smoothed_loss = avg_loss / (1 - beta**batch_num)
#         #Stop if the loss is exploding
#         if batch_num > 1 and smoothed_loss > 4 * best_loss:
#             return log_lrs, losses
#         #Record the best loss
#         if smoothed_loss < best_loss or batch_num==1:
#             best_loss = smoothed_loss
#         #Store the values
#         losses.append(smoothed_loss)
#         log_lrs.append(math.log10(lr))
#         #Do the SGD step
#         loss.backward()
#         optimizer.step()
#         #Update the lr for the next step
#         lr *= mult
#         optimizer.param_groups[0]['lr'] = lr
#     return log_lrs, losses


# import pandas as pd
# print(get_result(pd.read_csv('oof_df.csv')))
# print(label_list)
# 评价指标发生了变化
# np.mean([roc_auc_score(y_pred[:, i], y_true[:, i]) for i in range(11)])


if __name__ == '__main__':
    import pandas as pd

    res_df = pd.read_csv('./oof_df.csv')
    temp_df = res_df.copy()
    # right_score = temp_df['pred_CVC - Abnormal - Right Atrium'].tolist()
    # other_score = temp_df['pred_CVC - Abnormal - Other']
    # temp_df['pred_CVC - Abnormal'] = 
    new_score = []
    for step, row in temp_df.iterrows():
        ori_score = row['pred_CVC - Abnormal']
        max_score = ori_score
        # row['pred_CVC - Abnormal'] = max(max_score,max(row['pred_CVC - Abnormal - Right Atrium'], row['pred_CVC - Abnormal - Other']))
        # if step < 5:
        #     print('ori is {}'.format(ori_score))
        # row['pred_CVC - Abnormal'] = max(row['pred_CVC - Abnormal - Right Atrium'], row['pred_CVC - Abnormal - Other'])
        row['pred_CVC - Abnormal'] = (ori_score + row['pred_CVC - Abnormal - Right Atrium'] + row[
            'pred_CVC - Abnormal - Other']) / 3
        # if step < 5:
        #     print('mean score is {}'.format(row['pred_CVC - Abnormal']))
        #     print('right score is {}'.format(row['pred_CVC - Abnormal - Right Atrium']))
        #     print('other score is {}'.format(row['pred_CVC - Abnormal - Other']))
        #     print('post score is {}'.format(row['pred_CVC - Abnormal']))
        new_score.append(row['pred_CVC - Abnormal'])

    # new_folds = temp_df.copy()
    temp_df.drop('pred_CVC - Abnormal - Right Atrium', axis=1, inplace=True)
    temp_df.drop('pred_CVC - Abnormal - Other', axis=1, inplace=True)
    temp_df['pred_CVC - Abnormal'] = new_score
    # new_folds.to_csv('./data/test_res.csv',index=False)
    # temp_df.to_csv()
    # print(res_df['pred_CVC - Abnormal'][:2])
    # print(temp_df['pred_CVC - Abnormal'][:2])
    # new_folds = temp_df.copy()
    # print(new_folds)
    # print(new_folds.columns)
    print(get_result(res_df))
    print(get_result(temp_df))
