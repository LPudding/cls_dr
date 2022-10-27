from albumentations.augmentations.transforms import MultiplicativeNoise, OpticalDistortion
from torch.utils.data import Dataset
import pandas as pd 
import cv2
import os
import numpy as np 
import torch
import albumentations
import albumentations as A 
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, CenterCrop,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, HueSaturationValue,
    IAAAdditiveGaussianNoise, CoarseDropout, Transpose
    )
from albumentations.pytorch import ToTensorV2
# from data_process.process_utils import AdvancedLineAugmentation
# from utils.get_args import get_argparse

label_list = ['Nucleoplasm',
            'Nuclear membrane',
            'Nucleoli',
            'Nucleoli fibrillar center',
            'Nuclear speckles',
            'Nuclear bodies',
            'Endoplasmic reticulum',
            'Golgi apparatus',
            'Intermediate filaments',
            'Actin filaments',
            'Microtubules',
            'Mitotic spindle',
            'Centrosome',
            'Plasma membrane',
            'Mitochondria',
            'Aggresome',
            'Cytosol',
            'Vesicles and punctate cytosolic patterns',
            'Negative']

# 多分类
# class HPADataset(Dataset):
#     def __init__(self, df, path, transform=None, mode='train'):
#         self.path = path
#         self.df = df
#         print(self.df.head)
#         self.img_ids = self.df['image_id'].values
#         self.labels = self.df['image_labels'].values
#         # self.img_size = img_size        
#         self.transform = transform   
#         self.mode = mode 
#         self.cell_ids = self.df['cell_id'].values
        
#     def __len__(self):
#         return len(self.df) 
    
#     def __getitem__(self, index):
#         try:
#             img = cv2.imread(os.path.join(self.path, f'{self.img_ids[index]}_{self.cell_ids[index]}'+'.jpg'))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         except:
#             print(os.path.join(self.path, f'{self.img_ids[index]}_{self.cell_ids[index]}'+'.jpg'))
# #         print(img)
# #         img = get_rgby_image(self.path, self.img_ids[index])
#         # x = self._get_image(self.img_ids[index])
#         if self.transform:
#             augmented = self.transform(image=img)
#             img = augmented['image']
# #         print(img.shape) # 3,size,size
#         label = int(self.labels[index])
# #         label = np.array(y)
#         # img = torch.from_numpy(img, dtype=torch.float32)
#         label = torch.tensor(label)                               
#         # y = y.sum(axis=0)
#         if self.mode == 'test':
#             return img
#         else:
#             return img, label

# 多标签
class HPADataset(Dataset):
    def __init__(self, df, path, transform=None, mode='train'):
        self.path = path
        self.df = df
        print(self.df.head)
        self.img_ids = self.df['image_id'].values
        self.labels = self.df['image_labels'].values
        # self.img_size = img_size        
        self.transform = transform   
        self.mode = mode 
        self.cell_ids = self.df['cell_id'].values
        
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.path, f'{self.img_ids[index]}_{self.cell_ids[index]}'+'.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         print(img)
#         img = get_rgby_image(self.path, self.img_ids[index])
        # x = self._get_image(self.img_ids[index])
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
#         print(img.shape) # 3,size,size
        y = str(self.labels[index])
        y = y.split('|')
        y = list(map(int, y))            
        # y = np.eye(19, dtype='float')[y]
        label = [0 for _ in range(19)]
        for l in y:
            label[l] = 1
        label = np.array(label)
        # img = torch.from_numpy(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float)                               
        # y = y.sum(axis=0)
        if self.mode == 'test':
            return img
        else:
            return img, label

def get_rgby_image(path, id):

    R = cv2.imread(os.path.join(path, id, '_red.png'), cv2.IMREAD_UNCHANGED)
    Y = cv2.imread(os.path.join(path, id,'_yellow.png'), cv2.IMREAD_UNCHANGED)
    G = cv2.imread(os.path.join(path, id,'_green.png'), cv2.IMREAD_UNCHANGED)
    B = cv2.imread(os.path.join(path, id,'_blue.png'), cv2.IMREAD_UNCHANGED)
    print(R)
    print(G)
    print(B)
    print(Y)
    # R = R.astype(np.uint8)
    # Y = Y.astype(np.uint8)
    # G = G.astype(np.uint8)
    # B = B.astype(np.uint8)
    img = np.stack((
                R/2 + Y/2, 
                G/2 + Y/2, 
                B),-1)
    print(img)
    img = np.divide(img, 255)
    # img = np.stack([R,G,B,Y]).transpose(1,2,0)
    return img
# class HpaDefaultDataset(Dataset):
#     def __init__(self,
#                  dataset_dir,
#                  transform=None,
#                  idx_fold=0,
#                  num_fold=5,
#                  mode='train'
#                  ):
#         # self.split = split
#         self.idx_fold = idx_fold
#         self.num_fold = num_fold
#         self.transform = transform
#         self.dataset_dir = dataset_dir
#         # self.split_prefix = split_prefix
#         self.mode = mode
#         self.images_dir = os.path.join(dataset_dir, 'train')
#         self.external_images_dir = os.path.join(dataset_dir, 'external')

#         self.df_labels = self.load_labels()
#         self.examples = self.load_examples()
#         self.size = len(self.examples)

#     def load_labels(self):
#         labels_path = '{}.csv'.format(self.mode)
#         labels_path = os.path.join(self.dataset_dir, labels_path)
#         df_labels = pd.read_csv(labels_path)
#         # df_labels = df_labels[df_labels['Split'] == self.split]
#         df_labels = df_labels.reset_index()

#         train_id_len = len('770126a4-bbc6-11e8-b2bc-ac1f6b6435d0')
#         def to_filepath(v):
#             if len(v) == train_id_len:
#                 return os.path.join(self.images_dir, v + '.png')
#             else:
#                 return os.path.join(self.external_images_dir, v + '.png')

#         df_labels['filepath'] = df_labels['Id'].transform(to_filepath)
#         return df_labels

#     def load_examples(self):
#         return [(row['Id'], row['filepath'], [int(l) for l in row['Target'].split(' ')])
#                 for _, row in self.df_labels.iterrows()]

#     def __getitem__(self, index):
#         example = self.examples[index]

#         filename = example[1]
#         image = misc.imread(filename)

#         label = [0 for _ in range(28)]
#         for l in example[2]:
#             label[l] = 1
#         label = np.array(label)

#         if self.transform is not None:
#             image = self.transform(image)

#         return {'image': image,
#                 'label': label,
#                 'key': example[0]}

#     def __len__(self):
#         return self.size

# convert external data to RGBY images
# def image_read_4channel_external(path, name):
#     image_red = cv2.imread(path + '/%s_red.jpg' % name, 0)
#     image_yellow = cv2.imread(path + '/%s_yellow.jpg' % name, 0)
#     image_blue = cv2.imread(path + '/%s_blue.jpg' % name, 0)
#     image_green = cv2.imread(path + '/%s_green.jpg' % name, 0)
#     if image_yellow is None:
#         image_yellow = np.zeros_like(image_red)

#     image_red = cv2.normalize(image_red,None,0,255,cv2.NORM_MINMAX)
#     image_red = cv2.resize(image_red, (512, 512))
#     image_red = image_red.astype(np.uint8)

#     image_yellow = cv2.normalize(image_yellow,None,0,255,cv2.NORM_MINMAX)
#     image_yellow = cv2.resize(image_yellow, (512, 512))
#     image_yellow = image_yellow.astype(np.uint8)
    
#     image_blue = cv2.normalize(image_blue,None,0,255,cv2.NORM_MINMAX)
#     image_blue = cv2.resize(image_blue, (512, 512))
#     image_blue = image_blue.astype(np.uint8)
    
#     image_green = cv2.normalize(image_green,None,0,255,cv2.NORM_MINMAX)
#     image_green = cv2.resize(image_green, (512, 512))
#     image_green = image_green.astype(np.uint8)
#     image = np.stack([image_red,image_green,image_blue,image_yellow]).transpose(1,2,0)

#     return image

# convert kaggle data to RGBY images
# def image_read_4channel_kaggle(path, name):
#     image_red = cv2.imread(path + '/%s_red.png' % name, 0)
#     image_yellow = cv2.imread(path + '/%s_yellow.png' % name, 0)
#     image_blue = cv2.imread(path + '/%s_blue.png' % name, 0)
#     image_green = cv2.imread(path + '/%s_green.png' % name, 0)

#     image_red = image_red.astype(np.uint8)
#     image_yellow = image_yellow.astype(np.uint8)
#     image_blue = image_blue.astype(np.uint8)
#     image_green = image_green.astype(np.uint8)
#     image = np.stack([image_red,image_green,image_blue,image_yellow]).transpose(1,2,0)

#     return image

# class HPAIC_process_external(Dataset):
#     def __init__(self,
#                  name_list = None,
#                  transform = None,
#                  read_path = None,
#                  write_path = None
#                  ):
#         self.name_list = name_list
#         self.transform = transform
#         self.read_path = read_path
#         self.write_path = write_path

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, idx):
#         name = self.name_list[idx]

#         image = image_read_4channel_external(self.read_path, name)
#         cv2.imwrite(self.write_path + name + '.png', image)
#         return name

# class HPAIC_process_kaggle(Dataset):
#     def __init__(self,
#                  name_list = None,
#                  transform = None,
#                  read_path = None,
#                  write_path = None
#                  ):
#         self.name_list = name_list
#         self.transform = transform
#         self.read_path = read_path
#         self.write_path = write_path

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, idx):
#         name = self.name_list[idx]

#         image = image_read_4channel_kaggle(self.read_path, name)
#         cv2.imwrite(self.write_path + name + '.png', image)
#         return name



def get_transforms(args, mode):
    
    if mode == 'train':
        return Compose([
            RandomResizedCrop(args.image_size,args.image_size, scale=(0.9, 1.0)),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.2),
            # RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.2),
            albumentations.CLAHE(clip_limit=(1,4), p=0.5),
            # albumentations.OpticalDistortion(),
            # AdvancedLineAugmentation(p=0.1),
            # albumentations.OneOf([
            #     albumentations.OpticalDistortion(distort_limit=1.0),
            #     albumentations.ElasticTransform(alpha=3),
            # ], p=0.2),
            
            # Resize(512, 512),
            # CoarseDropout(p=0.2),
            # albumentations.JpegCompression(quality_lower=80, p=0.5),
            # albumentations.MultiplicativeNoise(p=0.2),
            # A.IAASharpen(p=0.5),
            # Cutout(max_h_size=16, max_w_size=16, num_holes=16, fill_value=(192, 192, 192), p=0.2),
            # Cutout(p=0.2, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif mode == 'val' or mode == 'test':
        return Compose([
            Resize(args.image_size, args.image_size),
            # CenterCrop(380,380, p=1.),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], 
            ),
            ToTensorV2(),
        ])



if __name__ == '__main__':
    df_data = pd.read_csv('../data/train_fold.csv')
    # print(df_data['ID'].values[0])
    data_dir = '../data/rgby/train_768/'
    my_dataset = HPADataset(data_dir, df_data)
    x,y = my_dataset[0]
    print(x.shape)
    print(y)
    # df_data.head()
    # annotation_data = pd.read_csv('../data/train_annotations.csv')
    # train_dataset = Ranzcr_dataset(df_data, '../data/train')
    # 判断有黑边图片的数量
    # cnt = 0
    # for i, row in df_data.iterrows():
    #     img_id = row['StudyInstanceUID']
    #     gray_img = cv2.imread(os.path.join('../data/train', img_id + '.jpg'), cv2.IMREAD_GRAYSCALE)
    #     mask = gray_img > 0
    #     process_img = gray_img[np.ix_(mask.any(1), mask.any(0))]
    #     if process_img.shape != gray_img.shape:
    #         cnt += 1
    # print(cnt)
    # print(type(train_dataset[0][0]))
    # print(train_dataset[0][0].shape)
    # from sklearn.model_selection import StratifiedKFold, KFold
    # skf = StratifiedKFold(n_splits=5, random_state=442)
    # for i, (train_index, val_index) in enumerate(skf.split(df_data['image_id'], df_data['label'])):
    #     print(len(train_index))
    #     print(len(val_index))
    