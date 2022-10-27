import cv2
import pandas as pd
import os

df = pd.read_csv("../train_fold-5.csv")
base_dir = "../../../../DataSets"
to_dir = "origin_image"
for i in range(df['imgnames'].size):
    imgname = df['imgnames'][i]
    # if imgname.split('/')[0] != 'Blindness_Detection_Images_2015&2019':
    #     continue
    # img = cv2.imread(os.path.join(base_dir, imgname))
    # img = cv2.resize(img, (512, 512))
    dataset_path = imgname.split("/")[0]
    name = imgname.split("/")[-1]
    # if not os.path.exists(os.path.join(to_dir, dataset_path)):
    #     os.makedirs(os.path.join(to_dir, dataset_path))
    # cv2.imwrite(os.path.join(to_dir, dataset_path, name), img)
    df.loc[i, 'imgnames'] = os.path.join(dataset_path, name)
    print(os.path.join(to_dir, dataset_path, name))
df.to_csv(os.path.join(to_dir, "fold-5.csv"))

