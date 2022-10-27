import pandas as pd 
from sklearn.model_selection import GroupKFold, StratifiedKFold
# from data_process.process_data import label_list
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np 
import random
import cv2
import os
import re


def extract_source(excel_name, index, split):
    df_source = pd.read_csv(excel_name)
    df_source['source'] = df_source['imgnames'].map(lambda x: x.split(split)[index])
    df_source.to_csv(excel_name)


# label_list = ['Nucleoplasm',
#             'Nuclear membrane',
#             'Nucleoli',
#             'Nucleoli fibrillar center',
#             'Nuclear speckles',
#             'Nuclear bodies',
#             'Endoplasmic reticulum',
#             'Golgi apparatus',
#             'Intermediate filaments',
#             'Actin filaments',
#             'Microtubules',
#             'Mitotic spindle',
#             'Centrosome',
#             'Plasma membrane',
#             'Mitochondria',
#             'Aggresome',
#             'Cytosol',
#             'Vesicles and punctate cytosolic patterns',
#             'Negative']


def split_k_fold(n_splits, shuffle, random_state, df_ori, label_list=None):
    df_fold = df_ori.copy()
    if label_list is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (train_ind, val_ind) in enumerate(skf.split(df_fold, df_fold['label'])):
            df_fold.loc[val_ind, 'fold'] = int(fold)
        df_fold['fold'] = df_fold['fold'].astype(int)
    else:
        skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (train_ind, val_ind) in enumerate(skf.split(df_fold, df_fold[label_list])):
            df_fold.loc[val_ind, 'fold'] = int(fold)
        df_fold['fold'] = df_fold['fold'].astype(int)
    return df_fold



# ori_df = pd.read_csv('../data/train.csv')
# new_df = ori_df.copy()
# new_label = []
# for i, row in ori_df.iterrows():
#     ori_label = row['Label']
#     ori_label = ori_label.split('|')
#     y = list(map(int, ori_label))
#     # y = np.eye(19, dtype='float')[y]
#     label = [0 for _ in range(19)]
#     for l in y:
#         label[l] = 1
#     new_label.append(label)

# new_df[label_list] = new_label
# new_df.to_csv('../data/new_train.csv', index=False)

# MultiLabel split fold
# df_data = pd.read_csv('../data/new_train.csv')
# folds = df_data.copy()
# skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for fold, (train_index, val_index) in enumerate(skf.split(df_data, df_data[label_list])):
#     folds.loc[val_index, 'fold'] = int(fold)
#
# folds['fold'] = folds['fold'].astype(int)
# folds.to_csv('../data/train_fold.csv', index=False)

# features:
# base_path
# image_name
# labels
# fold

# ill
# source

if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    skf1 = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Blindness Detection Images 2019
    train2019_path = "../../../../DataSets/Blindness_Detection_Images_2015&2019/"
    df_train2019 = pd.read_csv(os.path.join(train2019_path, 'labels', 'trainLabels19.csv'))
    df_train2019 = df_train2019.rename(columns={'diagnosis': 'level', 'id_code': 'image'})
    for fold, (train_index, val_index) in enumerate(skf.split(df_train2019, df_train2019['level'])):
        df_train2019.loc[val_index, 'fold'] = int(fold)
    df_train2019['fold'] = df_train2019['fold'].astype(int)
    df_train2019 = df_train2019[['image', 'level', 'fold']]
    df_train = df_train2019[df_train2019.fold != 0]
    df_test = df_train2019[df_train2019.fold == 0]
    print(df_train['level'].value_counts())
    print(df_test['level'].value_counts())
    df_train.to_csv('trainLabels19.csv', index=False)
    df_test.to_csv('testLabels19.csv', index=False)
    exit()

    # combine Messidor and FGADR to solve the problem of label
    # Messidor
    Messidor_path = "../../../../DataSets/Messidor/"
    messidor_label_path = os.listdir(os.path.join(Messidor_path, "labels"))
    # print(messidor_label_path)
    df_messidor = pd.DataFrame()
    for path in messidor_label_path:
        df_t = pd.read_excel(os.path.join(Messidor_path, 'labels', path), engine='xlrd')
        df_t['Image name'] = df_t['Image name'].map(lambda x: os.path.join(path[11:17], x))
        df_messidor = pd.concat([df_messidor, df_t], axis=0, ignore_index=True)
    df_messidor = df_messidor.rename(columns={'Retinopathy grade': 'label', 'Image name': 'image_name'})
    df_messidor['source'] = 'Messidor'
    df_messidor['image_name'] = df_messidor['image_name'].map(lambda x: x.split('/')[-1])
    for fold, (train_index, val_index) in enumerate(skf.split(df_messidor, df_messidor['label'])):
        df_messidor.loc[val_index, 'fold'] = int(fold)
    df_messidor = df_messidor[['image_name', 'label', 'source', 'fold']]
    df_messidor = df_messidor.reset_index(drop=True)
    df_messidor['fold'] = df_messidor['fold'].astype(int)
    print(df_messidor)
    # exit()

    # Messidir2
    Messidor2_path = "../../../../DataSets/Messidor-2/"
    Messidor2_label_path = os.path.join(Messidor2_path, "messidor-2-label.csv")
    df_messidor2 = pd.read_csv(Messidor2_label_path)
    df_messidor2 = df_messidor2.rename(columns={'image_id': 'image_name', 'adjudicated_dr_grade': 'label'})
    df_messidor2 = df_messidor2[['image_name', 'label']]
    df_messidor2['source'] = 'Messidor-2'
    df_messidor2['image_name'] = df_messidor2['image_name'].map(lambda x: x.split('/')[-1])
    df_messidor2['image_name'] = df_messidor2['image_name'].map(lambda x: x.split('.')[0] + '.JPG' if x.split('.')[1] == 'jpg' else x)
    df_messidor2 = df_messidor2[np.isnan(df_messidor2['label']) == 0]
    df_messidor2['label'] = df_messidor2['label'].map(lambda x: int(x))
    df_messidor2 = df_messidor2.reset_index(drop=True)
    for fold, (train_index, val_index) in enumerate(skf.split(df_messidor2, df_messidor2['label'])):
        df_messidor2.loc[val_index, 'fold'] = int(fold)
    df_messidor2 = df_messidor2[['image_name', 'label', 'source', 'fold']]
    df_messidor2['fold'] = df_messidor2['fold'].astype(int)
    print(df_messidor2)
    # exit(0)

    # FGADR
    FGADR_path = "../../../../DataSets/FGADR-Seg-set_Release/"
    df_FGADR = pd.read_csv(os.path.join(FGADR_path, 'Seg-set', 'DR_Seg_Grading_Label.csv'), header=None)
    df_FGADR.columns = ['image_name', 'label']
    df_FGADR['source'] = 'FGADR-Seg-set_Release'
    df_FGADR['image_name'] = df_FGADR['image_name'].map(lambda x: x.split('/')[-1])
    df_FGADR = df_FGADR.reset_index(drop=True)
    for fold, (train_index, val_index) in enumerate(skf.split(df_FGADR, df_FGADR['label'])):
        df_FGADR.loc[val_index, 'fold'] = int(fold)
    df_FGADR['fold'] = df_FGADR['fold'].astype(int)
    df_FGADR = df_FGADR[['image_name', 'label', 'source', 'fold']]
    print(df_FGADR)
    # exit(0)

    # DDR
    DDR_path = "../../../../DataSets/DDR dataset/DR_grading/"
    splits = ['train', 'valid', 'test']
    DDR_paths = [os.path.join(DDR_path, split + '.txt') for split in splits]
    df_DDR = pd.DataFrame(columns=['image_name', 'label'])

    for path in DDR_paths:
        with open(path, "r") as f:
            split = path.split('/')[-1].split('.')[0]
            while True:
                line = f.readline()
                if not line:
                    break
                # print(line)
                image_name = line.split(" ")[0]
                label = line.split(" ")[1][:1]
                df_DDR = df_DDR.append([{'image_name': image_name, 'label': label}], ignore_index=True)
    df_DDR['source'] = 'DDR dataset'
    # df_DDR['image_name'] = df_DDR['image_name'].map(lambda x: x.split('/')[-1])

    for fold, (train_index, val_index) in enumerate(skf.split(df_DDR, df_DDR['label'])):
        df_DDR.loc[val_index, 'fold'] = int(fold)
    df_DDR['fold'] = df_DDR['fold'].astype(int)
    df_DDR = df_DDR[['image_name', 'label', 'source', 'fold']]
    print(df_DDR)
    # exit()
    # exit(0)


    # Blindness Detection Images 2019
    train2019_path = "../../../../DataSets/Blindness_Detection_Images_2015&2019/"
    df_train2019 = pd.read_csv(os.path.join(train2019_path, 'labels', 'trainLabels19.csv'))
    df_train2019 = df_train2019.rename(columns={'diagnosis': 'label', 'id_code': 'image_name'})
    df_train2019['image_name'] = df_train2019['image_name'].map(lambda x: x+'.jpg')
    df_train2019['source'] = 'Blindness_Detection_Images_2019'

    for fold, (train_index, val_index) in enumerate(skf.split(df_train2019, df_train2019['label'])):
        df_train2019.loc[val_index, 'fold'] = int(fold)
    df_train2019['fold'] = df_train2019['fold'].astype(int)
    df_train2019 = df_train2019[['image_name', 'label', 'source', 'fold']]
    print(df_train2019)
    # exit()

    df = pd.concat([df_FGADR, df_messidor, df_messidor2, df_train2019, df_DDR], ignore_index=True, axis=0)
    # df = df_FGADR
    df['label'] = df['label'].map(lambda x: int(x))
    df['ill'] = df['label'].map(lambda x: 1 if x >= 1 else 0)
    df['base_dir'] = 'datasets/origin_image'
    df = df[df.label != 5].reset_index(drop=True)

    print(df['label'].value_counts())
    df.to_csv('../fold-5.csv', index=False)




