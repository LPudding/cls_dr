import cv2
import os
import numpy as np
import pandas as pd
from cls_train import TCDataset, get_transforms
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from tqdm import tqdm

from models.model import ResNet
from models.wsol.resnet import ResNet50WSOL, ResNet18Adl
from utils.utils import Mean_calc
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, CenterCrop,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, HueSaturationValue,
    IAAAdditiveGaussianNoise, CoarseDropout, Transpose
)
from albumentations.pytorch import ToTensorV2

gpus = True
device_ids = [0, 1]  # 可用GPU
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

area_cal = Mean_calc()

fmap_block = list()
grad_block = list()


# 图片预处理
def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]  # 1
    img = np.ascontiguousarray(img)  # 2
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(512),
        # CenterCrop(380,380, p=1.),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # 3
    return img


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    if len(grad_block):
        grad_block.pop()
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    if len(fmap_block):
        fmap_block.pop()
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(imgs, feature_maps, grads, out_dir, img_path, output, pred='pred'):
    H, W, _ = imgs.shape
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_maps[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    # cam = np.array(cam)
    # cam = (cam - cam.min()) / (cam.max() - cam.min())
    # print(cam)
    # print(cam.max())
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))
    # cv2.imshow('s', cam)
    # cv2.waitKey()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * imgs

    # print(exit(0))
    base_dir = os.path.join(out_dir, "cam", img_path.split('/')[-2], pred)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    output_str = str(output[0]) + '_' + str(int(output[1]))
    img_name = img_path.split('/')[-1].split('.')[0]
    img_suffix = '.' + img_path.split('/')[-1].split('.')[1]
    path_cam_img = os.path.join(base_dir, img_name + '-' + output_str + img_suffix)
    print("output_path:{}".format(path_cam_img))
    # exit(0)
    cv2.imwrite(path_cam_img, cam_img.astype(np.uint8))


def update(up, down, left, right, h1, h2, w1, w2):
    if h1 < up:
        up = h1
    if h2 > down:
        down = h2
    if w1 < left:
        left = w1
    if w2 > right:
        right = w2
    return up, down, left, right


def takeFifth(elem):
    return elem[4]


# bounder = [up, down, left, right]
def pick_pixels(img, cam, threshold, center_threshold):
    retVal, cam_m = cv2.threshold(np.uint8(255 * cam), np.uint8(threshold * 255), 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(cam_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box = []
    # show the pictures
    # cv2.imshow("show", img)
    # cv2.waitKey(0)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        heat_avg = Mean_calc()
        black_avg = Mean_calc()
        flag = 1
        for x1 in range(x, x + w):
            for y1 in range(y, y + h):
                heat_avg.update(cam[y1][x1])
                if img[y1][x1][0] < 10 and img[y1][x1][1] < 10 and img[y1][x1][2] < 10:
                    black_avg.update(1)
                else:
                    black_avg.update(0)
                if flag and cam[y1][x1] > center_threshold:
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    area_cal.update(w*h)
                    flag = 0
        if flag == 0 and black_avg.get_mean() < 0.5:
            bounding_box.append([x, y, w, h, heat_avg.get_mean()])
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    bounding_box.sort(key=takeFifth, reverse=True)
    print(bounding_box)
    # cv2.imshow("show", img)
    # cv2.waitKey(0)

    return bounding_box


def split_patches(img, feature_maps, grads, output, out_dir, img_path, patch_size=64):
    H, W, _ = img.shape
    img_copy = img.copy()
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_maps[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))
    cam_matrix = cam.copy()
    bounding_box = pick_pixels(img_copy, cam_matrix, 0.5, 0.8)

    bounding_dir = os.path.join(out_dir, "bounding_box", img_path.split('/')[-2])
    if not os.path.exists(bounding_dir):
        os.makedirs(bounding_dir)
    patch_dir = os.path.join(out_dir, "lesion_patches", img_path.split('/')[-2])
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    output_str = str(output[0]) + '_' + str(output[1])
    img_name = img_path.split('/')[-1].split('.')[0]
    img_suffix = '.' + img_path.split('/')[-1].split('.')[1]
    # print("output_path:{}".format(path_bounding_img))
    count = 0
    for index, box in enumerate(bounding_box):
        x, y, w, h, avg = box
        x_center = int(x + w / 2)
        y_center = int(y + h / 2)
        # over cover or outbound
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if index < 2:
            patch = img[y:y + h, x:x + w, :]
            patch = cv2.resize(patch, (64, 64))
            count += 1
            path_patch_img = os.path.join(patch_dir, img_name + '-' + str(index) + img_suffix)
            cv2.imwrite(path_patch_img, patch)
    path_bounding_img = os.path.join(bounding_dir, img_name + '-' + output_str + img_suffix)
    cv2.imwrite(path_bounding_img, img_copy)
    return count



# def gen_cam(image_path, label, index, net, output_dir):
#     print("input_path:{}".format(image_path))
#     img = cv2.imread(image_path, 1)
#     img_input = img_preprocess(img)
#     img_input = img_input.to(device)
#     output, _ = net(img_input)
#     # print(output)
#     idx = np.argmax(output.cpu().data.numpy())
#     # print("predict: {}".format(classes[idx]))
#     # backward
#     net.zero_grad()
#
#     if index != -1:
#         pred = index
#     else:
#         pred = idx
#     # for predcit class
#     class_loss = output[0, pred]
#     # print(class_loss)
#     class_loss.backward()
#
#     # 生成cam
#     index = 0
#     grads_val = grad_block[index].cpu().data.numpy().squeeze()
#     fmap = fmap_block[index].cpu().data.numpy().squeeze()
#     # 保存cam图片
#     saved_path = image_path.split('/')[-1]
#     # cam_show_img(img, fmap, grads_val, output_dir, saved_path)
#     return pd.DataFrame([{'image': saved_path, 'ground true': label, 'predict': idx}])


def get_grad(image_path, label, net, index=5):
    print("input_path:{}".format(image_path))
    img = cv2.imread(image_path, 1)
    img_input = img_preprocess(img)
    img_input = img_input.to(device)
    output, _ = net(img_input)
    # print(output)
    # idx = np.argmax(output.cpu().data.numpy())
    # print("predict: {}".format(classes[idx]))
    # print(output.squeeze(0).sigmoid().cpu().data.numpy())
    if output.squeeze(0).sigmoid().cpu().data.numpy()[index] >= 0.5:
        idx = 1
    else:
        idx = 0
    # backward
    net.zero_grad()

    # if index != -1:
    #     pred = index
    # else:
    #     pred = idx
    # for predcit class
    class_loss = output[0, index]
    # print(class_loss)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    # 保存cam图片
    saved_path = image_path.split('/')[-1]
    return img, fmap, grads_val, pd.DataFrame([{'image': saved_path, 'ground_true': label, 'predict': idx}]), output


def cam_viewer(model, val_df):
    # model.layer4[1].conv2.register_forward_hook(farward_hook)
    # model.layer4[1].conv2.register_backward_hook(backward_hook)
    model.feature[7][2].conv3.register_forward_hook(farward_hook)  # 9
    model.feature[7][2].conv3.register_backward_hook(backward_hook)
    # model.layer4[2].conv3.register_forward_hook(farward_hook)  # 9
    # model.layer4[2].conv3.register_backward_hook(backward_hook)

    df_result = pd.DataFrame(columns=['image', 'ground_true', 'predict'])
    df_patch = pd.DataFrame(columns=['image_name', 'label', 'source'])
    for i in range(len(val_df)):
        # if val_df['image_name'][i] != '0007_2.png':
        #     continue
        print(val_df['image_name'][i])
        df_i = val_df.iloc[i]
        fold = df_i['fold']
        img_name = val_df['image_name'][i]
        prefix = img_name.split('.')[0]
        suffix = img_name.split('.')[1]
        img_path = os.path.join('..', df_i['base_dir'], df_i['source'], df_i['image_name'])
        img, fmap, grads_val, df_t, output = get_grad(img_path, df_i['NV'], model, )
        df_result = df_result.append(df_t, ignore_index=True)
        if fold == 0:
            cam_show_img(img, fmap, grads_val, output_dir, img_path, [df_t.predict[0], df_t.ground_true[0]], 'NV')
        # for patches
        # if df_t.loc[0, 'ground_true'] > 1 and df_t.loc[0, 'predict'] == 1 and fold != 0:
        #     #     # print(torch.nn.functional.softmax(output))
        #     count = split_patches(img, fmap, grads_val, [df_t.predict[0], df_t.ground_true[0]], output_dir, img_path)
        #     for j in range(count):
        #         df_patch_t = pd.DataFrame([{'image_name': prefix + '-' + str(j) + '.' + suffix, 'label': df_i['label'],
        #                                    'source': df_i['source']}])
        #         df_patch = df_patch.append(df_patch_t)
                # print(df_patch)
                # exit(0)

    # df_patch['base_dir'] = os.path.join(output_dir, 'lesion_patches')
    # df_patch = df_patch[['image_name', 'label', 'source', 'base_dir']]
    # df_patch['ill'] = df_patch['label'].map(lambda x: 1 if x >= 1 else 0)
    # df_patch.to_csv(os.path.join(output_dir, 'patch.csv'))


def patch_cls_viewer(model, val_df, patch_size=64, image_size=512, is_split=False):
    if not is_split:
        val_df = val_df.dropna(subset=['patch_id'])
    acc_cal = Mean_calc()
    recall_cal = Mean_calc()
    # for split
    df_patch = pd.DataFrame(columns=['image_name', 'label', 'source', 'base_dir'])
    for i in range(len(val_df)):
        df_i = val_df.iloc[i]
        if not is_split:
            patch_list_t = df_i['patch_id'].split(',')
            patch_list_t = patch_list_t[:-1]
            patch_list_t = [int(x) for x in patch_list_t]
        source = df_i['source']
        print(df_i['image_name'])
        image_name = df_i['image_name'].split('/')[-1]
        img_path = os.path.join('../datasets/origin_image', df_i['source'], df_i['image_name'])
        image = cv2.imread(img_path, 1)
        image_copy = image.copy()
        # print(image.shape)
        image_input = img_preprocess(image)
        image_input = image_input.to(device)
        # print(image_input.shape)
        # exit()
        patch_num_row = int(image_size / patch_size)
        patch_list_p = []

        # for split
        prefix = df_i['image_name'].split('.')[0]
        suffix = df_i['image_name'].split('.')[1]

        for j in range(patch_num_row * patch_num_row):
            y_i = int(j / patch_num_row)
            y_i *= patch_size
            x_i = int(j % patch_num_row)
            x_i *= patch_size
            patch = image_input[:, :, y_i:y_i+patch_size, x_i:x_i+patch_size]
            # print(patch.shape)
            out, _ = model(patch)
            if np.argmax(out.cpu().data.numpy()) == 1:
                patch_list_p.append(j)
                cv2.rectangle(image_copy, (x_i+1, y_i+1), (x_i+patch_size-1, y_i+patch_size-1), (0, 0, 255))
                if is_split:
                    patch_name = prefix + '-' + str(j) + '.' + suffix
                    base_dir = os.path.join(output_dir, 'lesion_patches')
                    output_path = os.path.join(base_dir, df_i['source'])
                    df_patch = df_patch.append(pd.DataFrame([{'image_name': patch_name, 'label': df_i['label'],
                                                              'source': df_i['source'], 'base_dir': base_dir}]))
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    # print(os.path.join(output_path, patch_name))
                    cv2.imwrite(os.path.join(output_path, patch_name), image[y_i:y_i+patch_size, x_i:x_i+patch_size, :])
            else:
                cv2.rectangle(image_copy, (x_i+1, y_i+1), (x_i+patch_size-1, y_i+patch_size-1), (0, 255, 0))
        output_path = os.path.join(output_dir, "patch_viewer", source)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, image_name), image_copy)
        # exit()
        if not is_split:
            for patch_id in patch_list_p:
                if patch_id in patch_list_t:
                    acc_cal.update(1)
                else:
                    acc_cal.update(0)

            for patch_id in patch_list_t:
                if patch_id in patch_list_p:
                    recall_cal.update(1)
                else:
                    recall_cal.update(0)
    if is_split:
        df_patch.to_csv(os.path.join(output_dir, 'patch.csv'))
    else:
        acc = acc_cal.get_mean()
        recall = recall_cal.get_mean()
        f1 = 2 / (1 / acc + 1 / recall)
        print("parch count:{}".format(acc_cal.count))
        print("patch acc:{}".format(acc))
        print("patch recall:{}".format(recall))
        print("patch f1:{}".format(f1))
    # exit(0)


if __name__ == '__main__':

    # output_dir = 'r50_adl_0.1_0.8_cls_80e_32bs_labelsmooth/'
    # output_dir = 'cam_r50_adl_0.25_0.9_cls_80e_32bs_labelsmooth/'
    output_dir = 'cam_r50_multi_cls_80e_32bs_labelsmooth/'
    # extract label name
    # json_path = 'dr_2_labels.json'
    # with open(json_path, 'r') as load_f:
    #     load_json = json.load(load_f)
    # classes = {int(key): value for (key, value)
    #            in load_json.items()}
    # 只取标签名
    # classes = list(classes.get(key) for key in range(len(classes)))

    # 加载  预训练模型
    # net = ResNet18Adl(num_classes=2, adl_drop_rate=0.25, adl_drop_threshold=0.9)
    # net = ResNet50Adl(num_classes=2, adl_drop_rate=0.25, adl_drop_threshold=0.9)
    # net = ResNet50WSOL(dhl_importance_rate=0.6, dhl_drop_or_highlight_rate=0.85, dhl_drop_threshold=0.8,
    #                    dhl_highlight_threshold=0.5, num_classes=2, wsol='dhl', insert_position=[[], [], [], [], [2]])
    # net = ResNet50WSOL(num_classes=2, adl_drop_rate=0.25, adl_drop_threshold=0.9, wsol='adl',
    #                    insert_position=[[], [], [], [], [2]])
    net = ResNet(model_name='resnet50', pretrained=True, n_class=6)
    # pthfile = os.path.join('..', 'models', 'r50_dhl_0.6_0.85_0.8_0.5_cls_80e_32bs_labelsmooth', 'epoch_72_best.pth')
    # pthfile = os.path.join('..', 'models', 'r50_patch_cls_80e_2048bs_labelsmooth', 'epoch_23_best.pth')
    pthfile = os.path.join('..', 'models', 'r50_multi_cls_80e_32bs_labelsmooth', 'epoch_36_best.pth')
    # pthfile = os.path.join('..', 'models', 'r50_adl_0.25_0.9_cls_80e_32bs_labelsmooth',
    #                        'epoch_76_best.pth')
    net.load_state_dict(torch.load(pthfile)['state_dict'])
    net.cuda()
    net.eval()  # 8t_index(drop=True)

    # 注册hoofrom sklearn.datasets import load_digitsk
    print(net)

    # read the data all or part data?
    df_data = pd.read_csv('../fold-5-attr.csv')
    # df_data = pd.read_csv('../test_patch.csv')
    # df_data = pd.read_csv(os.path.join(output_dir, 'patch_all.csv'))
    df_data = df_data[df_data.source == 'FGADR-Seg-set_Release']
    # df_data = df_data[df_data.label >= 1]
    print(df_data['MA'].value_counts())
    val_df_data = df_data[df_data.fold == 0].reset_index(drop=True)
    # val_df_data = df_data
    print(len(val_df_data))
    # exit(0)

    cam_viewer(net, val_df_data)
    # patch_cls_viewer(net, val_df_data)
