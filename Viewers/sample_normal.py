import os.path
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random
import cv2

from utils.utils import Mean_calc


def extract_normal(image_name, patch_num, patch_size=64, threshold=0.5):
    img = cv2.imread(image_name)
    h, w, _ = img.shape
    p_list = []
    ind = 0
    while ind < patch_num:
        # print(ind)
        # random.seed()
        # print(w-patch_size)
        x = random.randint(0, w-patch_size)
        y = random.randint(0, h-patch_size)
        # print('{}/{}'.format(x, y))
        patch = img[y:y+patch_size, x:x+patch_size, :]
        # black_avg = Mean_calc()
        # for y_i in range(patch_size):
        #     for x_i in range(patch_size):
        #         dot = patch[y_i, x_i, :]
        #         if dot[0] < 5 and dot[1] < 5 and dot[2] < 5:
        #             black_avg.update(1)
        #         else:
        #             black_avg.update(0)
        # if black_avg.get_mean() <= 0.5:
        #     # cv2.imshow("show", patch)
        #     # cv2.waitKey(0)
        ind += 1
        p_list.append(patch)
    return p_list


if __name__ == '__main__':
    #
    output_dir = 'cam_r50_adl_0.25_0.9_cls_80e_32bs_labelsmooth/'

    df_patch = pd.read_csv(os.path.join(output_dir, 'patch.csv'))
    patches_num = df_patch.shape[0]
    print(patches_num)
    patches_every_pic = 10
    pic_num = patches_num / patches_every_pic

    df_data = pd.read_csv('../fold-5.csv')
    df_data = df_data[df_data.fold != 0]
    df_data = df_data[df_data.label == 0]
    df_data = df_data.reset_index(drop=True)
    # print(df_data.shape[0])
    # exit(0)
    # df_data = df_data[df_data.source != 'Blindness_Detection_Images_2015&2019'].reset_index(drop=True)
    df_row_size = df_data.shape[0]
    print(df_row_size)
    random.seed(42)
    random_list = random.sample(range(df_row_size), int(pic_num))
    print(random_list)

    df_patch = df_patch[['image_name', 'label', 'base_dir', 'source']]
    for index in random_list:
        df_i = df_data.iloc[index]

        image_path = os.path.join('..', df_i['base_dir'], df_i['source'], df_i['image_name'])
        print(image_path)
        patch_list = extract_normal(image_path, patches_every_pic)
        base_dir = os.path.join(output_dir, 'normal_patches', df_i['source'])
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        prefix = df_i['image_name'].split('.')[0]
        suffix = df_i['image_name'].split('.')[1]
        for i, patch in enumerate(patch_list):
            patch_name = prefix + '-' + str(i) + '.' + suffix
            save_name = os.path.join(base_dir, patch_name)
            print(save_name)
            cv2.imwrite(save_name, patch)
            df_t = pd.DataFrame([{'image_name': patch_name, 'label': 0, 'ill': 0, 'base_dir':
                                os.path.join(output_dir, 'normal_patches'), 'source': df_i['source']}])
            df_patch = df_patch.append(df_t)
    df_patch = df_patch.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(df_patch, df_patch['label'])):
        df_patch.loc[val_index, 'fold'] = int(fold)
    df_patch['fold'] = df_patch['fold'].astype(int)
    df_patch['ill'] = df_patch['label'].map(lambda x: 1 if x >= 1 else 0)
    df_patch.to_csv(os.path.join(output_dir, "patch_all.csv"))







