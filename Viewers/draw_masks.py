import numpy as np
import pandas as pd
import os
import cv2
import torchvision.transforms as transforms
from data_process.process_utils import split_k_fold


def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]  # 1
    img = np.ascontiguousarray(img)  # 2
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(512),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # 3
    return img


def check_boundary(mask_matrix, row, col, height, width):
    if row == 0 or row == height - 1:
        return True
    if col == 0 or col == width - 1:
        return True
    check_list = [[row, col-1], [row, col+1], [row+1, col], [row+1, col-1], [row+1, col+1], [row-1, col],
                  [row-1, col+1], [row, col-1]]
    for dot in check_list:
        if mask_matrix[dot[0], dot[1]] == 0:
            return True


def search_mask_boundary(mask_path):
    index = []
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
        height = mask.shape[0]
        width = mask.shape[1]
        for row in range(height):
            for col in range(width):
                # edge
                if mask[row][col] != 0 and check_boundary(mask, row, col, height, width):
                    index += [[row, col]]
    return index


def record_mask(base_dir, img_name, patch_size=32 , image_size=512):
    patch_set = set()
    patch_num_row = image_size / patch_size
    origin_img = cv2.imread(os.path.join(base_dir, 'Original_Images', img_name))
    print(os.path.join(base_dir, 'Original_Images', img_name))
    resize_img = cv2.resize(origin_img, (512, 512))
    # 8CC152 37BC9B 4A89DC 967ADC D770AD 434A54
    # GRASS MINT BLUE_JEANS LAVANDER PINK_ROSE DARK_GRAY
    HardExudate_Masks_index = search_mask_boundary(os.path.join(base_dir, "HardExudate_Masks", img_name))
    Hemohedge_Masks_index = search_mask_boundary(os.path.join(base_dir, 'Hemohedge_Masks', img_name))
    IRMA_Masks_index = search_mask_boundary(os.path.join(base_dir, 'IRMA_Masks', img_name))
    Microaneurysms_Masks_index = search_mask_boundary(os.path.join(base_dir, 'Microaneurysms_Masks', img_name))
    Neovascularization_Masks_index = search_mask_boundary(os.path.join(base_dir, 'Neovascularization_Masks', img_name))
    SoftExudate_Masks_index = search_mask_boundary(os.path.join(base_dir, 'SoftExudate_Masks', img_name))
    mask_list = [HardExudate_Masks_index, Hemohedge_Masks_index, IRMA_Masks_index, Microaneurysms_Masks_index,
                 Neovascularization_Masks_index, SoftExudate_Masks_index]
    # print(origin_img.shape)
    for i in range(len(mask_list)):
        mask = mask_list[i]
        for y, x in mask:
            y_i = int(y / patch_size)
            x_i = int(x / patch_size)
            patch_id = int(y_i * patch_num_row + x_i % patch_num_row)
            patch_set.add(patch_id)
    patch_list = list(patch_set)
    patch_list.sort()
    patch_str = ""
    for patch_id in patch_list:
        patch_str += str(patch_id)
        patch_str += ','
    return patch_str


# draw mask for FGADR
def draw_mask(base_dir, img_name):
    origin_img = cv2.imread(os.path.join(base_dir, 'Original_Images', img_name))
    print(os.path.join(base_dir, 'Original_Images', img_name))
    resize_img = cv2.resize(origin_img, (512, 512))
    # 8CC152 37BC9B 4A89DC 967ADC D770AD 434A54
    # GRASS MINT BLUE_JEANS LAVANDER PINK_ROSE DARK_GRAY
    RGB_list = [[140, 193, 82], [55, 188, 155], [74, 137, 220], [150, 122, 220], [215, 112, 173], [67, 74, 84]]
    HardExudate_Masks_index = search_mask_boundary(os.path.join(base_dir, "HardExudate_Masks", img_name))
    Hemohedge_Masks_index = search_mask_boundary(os.path.join(base_dir, 'Hemohedge_Masks', img_name))
    IRMA_Masks_index = search_mask_boundary(os.path.join(base_dir, 'IRMA_Masks', img_name))
    Microaneurysms_Masks_index = search_mask_boundary(os.path.join(base_dir, 'Microaneurysms_Masks', img_name))
    Neovascularization_Masks_index = search_mask_boundary(os.path.join(base_dir, 'Neovascularization_Masks', img_name))
    SoftExudate_Masks_index = search_mask_boundary(os.path.join(base_dir, 'SoftExudate_Masks', img_name))
    mask_list = [HardExudate_Masks_index, Hemohedge_Masks_index, IRMA_Masks_index, Microaneurysms_Masks_index,
                 Neovascularization_Masks_index, SoftExudate_Masks_index]
    # print(origin_img.shape)

    # save masks
    # for i in range(len(mask_list)):
    #     mask = mask_list[i]
    #     RGB = RGB_list[i]
    #     for index in mask:
    #         #BGR
    #         resize_img[index[0], index[1], 2] = RGB[0]
    #         resize_img[index[0], index[1], 1] = RGB[1]
    #         resize_img[index[0], index[1], 0] = RGB[2]
    # mask_img = cv2.resize(resize_img, (512, 512))
    # if not os.path.exists('mask'):
    #     os.makedirs('mask')
    #
    # cv2.imwrite("mask/{}".format(img_name), mask_img)

    mask_exist = [1 if len(x) > 0 else 0 for x in mask_list]
    return mask_exist


if __name__ == '__main__':
    df_data = pd.read_csv('../fold-5.csv')
    df_attr = pd.read_csv('../fold-5-attr.csv')
    FGADR_dir = os.path.join('../../../../DataSets/FGADR-Seg-set_Release/Seg-set/')

    # val_df_data = df_data[df_data.fold == 0].reset_index(drop=True)
    val_df_data = df_data
    val_df_data = val_df_data[val_df_data.source == 'FGADR-Seg-set_Release']
    df_attr['patch_id'] = ''

    # record attribute
    # df_attr = pd.DataFrame(columns=['image_name', 'base_dir', 'source', ''])
    for index in range(val_df_data.shape[0]):
        df_i = val_df_data.iloc[index]
        image_name = df_i['image_name']
        source = df_i['source']
        base_dir = df_i['base_dir']
        # if index == 10:
        #     break

        # record patch_id
        df_attr.loc[index, 'patch_id'] = record_mask(FGADR_dir, df_i['image_name'])
        print(df_attr.loc[index, 'patch_id'])

        # draw masks
        # draw_mask(FGADR_dir, df_i['image_name'])

        # record attribute and split
        # mask_exist = draw_mask(FGADR_dir, df_i['image_name'])
        #
        # df_attr = df_attr.append(pd.DataFrame([{'image_name': image_name, 'base_dir': base_dir, 'source': source,
        #                                         'MA': int(mask_exist[3]), 'HE': int(mask_exist[1]),
        #                                         'SE': int(mask_exist[5]), 'EX': int(mask_exist[0]),
        #                                         'IRMA': int(mask_exist[2]), 'NV': int(mask_exist[4])}]))
    # print(df_attr.reset_index(drop=True))
    # df_attr = df_attr.reset_index(drop=True)
    # label_list = ['MA', 'HE', 'SE', 'EX', 'IRMA', 'NV']
    # df_attr = split_k_fold(n_splits=5, shuffle=True, random_state=42, df_ori=df_attr, label_list=label_list)
    # df_attr = df_attr[['image_name', 'base_dir', 'source', 'MA', 'HE', 'SE', 'EX', 'IRMA', 'NV', 'fold']]
    #
    df_attr.to_csv('../fold-5-attr.csv')
    # df_data.to_csv('../fold-5.csv')
