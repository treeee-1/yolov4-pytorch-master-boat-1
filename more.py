# import cv2
# from glob2 import glob
#
# for fn in glob('*.jpg'): #确认文件格式
#     img=cv2.imread(fn)
#     horizontal_img=cv2.flip(img,1)
#     splitName=fn.split(".")
#     newName=splitName[0]
#     cv2.imwrite(newName+'_flip.jpg',horizontal_img)
#
import numpy as np
import cv2
import os

# 调整最大值
MAX_VALUE = 100


def update(input_img_path, output_img_path, lightness, saturation):
    """
    用于修改图片的亮度和饱和度
    :param input_img_path: 图片路径
    :param output_img_path: 输出图片路径
    :param lightness: 亮度
    :param saturation: 饱和度
    """

    # 加载图片 读取彩色图像归一化且转换为浮点型
    image = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # 1.调整亮度（线性变换)
    hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # 饱和度
    hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)
    cv2.imwrite(output_img_path, lsImg)


input_img_path = r'I:\antenna\flip\color'
output_img_path = r'I:\antenna\flip\colorover'

# 这里调参！！！
lightness = int(input("lightness(亮度-100~+100):"))  # 亮度
saturation = int(input("saturation(饱和度-100~+100):"))  # 饱和度

# 获得需要转化的图片路径并生成目标路径
image_filenames = [(os.path.join(input_img_path, x), os.path.join(output_img_path, x))
                   for x in os.listdir(input_img_path)]
# 转化所有图片
for path in image_filenames:

    update(path[0], path[1], lightness, saturation)
