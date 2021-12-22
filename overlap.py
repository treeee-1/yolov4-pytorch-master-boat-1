import os
import cv2
import numpy as np
def crop_one_picture(path, filename, cols, rows, overlap_cols, overlap_rows):
    big_img = cv2.imread(path + filename, 1)
    sum_rows = big_img.shape[0]  # 高度
    sum_cols = big_img.shape[1]  # 宽度
    save_path = path + "crop{0}_{1}/".format(cols, rows)  # 保存的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, i_start in enumerate(range(0, sum_cols , cols - overlap_cols)): #宽
        for j, j_start in enumerate(range(0, sum_rows , rows - overlap_rows)): #高
            i_end = i_start + cols #重叠度 = cols - 步长
            j_end = j_start + rows
            img = big_img[j_start: j_end, i_start: i_end, :]
            filenames = save_path + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + os.path.splitext(filename)[1]
            if img.shape[0] == rows and img.shape[1] == cols:
                cv2.imwrite(filenames, img)
            elif img.shape[0] == rows and img.shape[1] <= cols:
                img = np.pad(img, ((0, 0),(0, cols - img.shape[1]),(0, 0)), 'constant', constant_values=0)
                cv2.imwrite(filenames, img)
            elif img.shape[0] <= rows and img.shape[1] == cols:
                img = np.pad(img, ((0, rows - img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
                cv2.imwrite(filenames, img)
            elif img.shape[0] <= rows and img.shape[1] <= cols:
                img = np.pad(img, ((0, rows - img.shape[0]), (0, cols - img.shape[1]), (0, 0)), 'constant', constant_values=0)
                cv2.imwrite(filenames, img)

path = 'tif/'  # 要裁剪的图片所在的文件夹
filename = 'shuikucoord.tif'  # 要裁剪的图片名
cols = 880  # 小图片的宽度（列数）
rows = 650  # 小图片的高度（行数）
overlap_cols = 80
overlap_rows = 50
crop_one_picture(path, filename, cols, rows, overlap_cols, overlap_rows)