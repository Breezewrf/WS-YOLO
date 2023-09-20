import numpy as np
import cv2
import json
import os


# 定义三种颜色类别的RGB值
color_dict = {
    (107, 110, 207): 0,    # 类别1
    (82, 84, 163): 1,    # 类别2
    (57, 59, 121): 2     # 类别3
}

color_object_dict = {
    (57, 59, 121): 0,    # Clip applier
    (82, 84, 163): 1,    # Clip applier
    (107, 110, 207): 2,
    (156, 158, 222): 3,  # Hook
    (99, 121, 57): 4,    # Bipolar Forceps
    (140, 162, 82): 5,   # Bipolar Forceps
    (181, 207, 107): 6,  # Hook
    (206, 219, 156): 7,  # Vessel sealer
    (140, 109, 49): 8,   # Laparoscopic Scissors
    (189, 158, 57): 9,   # Monopolar Curved Scissors
    (231, 186, 82): 10,  # Stapler
    (231, 203, 148): 11, # Stapler
    (132, 60, 57): 12,   # Needle Driver
    (173, 73, 74): 13,   # Needle Driver
    (214, 97, 107): 14,  # Needle Driver
    (231, 150, 156): 15, # 
    (123, 65, 115): 16,
    (165, 81, 148): 17,  # Cadiere Forceps
    (206, 109, 189): 18, # 
    (222, 158, 214): 19,
    (222, 158, 214): 20,
    (214, 174, 107): 21,
    (158, 202, 225): 22,
    (198, 219, 239): 23,
    (198, 219, 239): 24,
    (253, 141, 60): 25,  # Tip-Up Fenestrated Grasper
    (253, 174, 107): 26, 
    (253, 208, 162): 27, # Liver retractor
    (49, 163, 84): 28,   # Suction
    (116, 196, 118): 29,
    (161, 217, 155): 30,
    (199, 233, 192): 31
}

# 指定train文件夹路径
train_folder = "/mnt/shared/wrf/simds_dataset/test_obj"

# 指定images文件夹路径
images_folder = os.path.join(train_folder, "masks")

# 遍历images文件夹，获取所有图片的路径
image_paths = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder) if filename.endswith(".jpg") or filename.endswith(".png")]


def annotate(im_path):
    # 读取mask图像
    mask_img = cv2.imread(im_path)
    # mask_img = cv2.imread("/mnt/shared/wrf/simds_dataset/test_obj/masks/exvivo_laparoscopic_01_image0015.png")
    # im_type = im_path.split('/')[-2]
    # assert im_type in ['masks', 'images'], f"path error: {im_path}"
    print(mask_img.shape)
    a = []
    b = []
    for i in mask_img:
        for j in i:
            if sum(j) not in a:
                a.append(sum(j))
                b.append(j)
    print(b)

    # 获取图像大小
    img_height, img_width, _ = mask_img.shape

    # 初始化YOLO目标检测数据集格式
    yolo_dataset = []

    # 遍历每个像素，提取边界框信息
    for color, class_id in color_object_dict.items():
        # 将特定颜色的像素赋值为1，其他像素赋值为0
        binary_mask = np.all(mask_img == np.array(color), axis=-1).astype(np.uint8)
        # 查找连通域，得到每个物体的轮廓
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 遍历每个轮廓，生成对应的COCO目标检测数据集格式
        for contour in contours:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            # 将边界框坐标转换为归一化坐标
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            # 将边界框信息添加到YOLO目标检测数据集格式
            yolo_dataset.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 将YOLO目标检测数据集格式保存为txt文件
    print(f"write txt into {im_path.replace('png', 'txt').replace('masks', 'labels')}")
    with open(im_path.replace('png', 'txt').replace('masks', 'labels'), "w") as f:
        f.write("\n".join(yolo_dataset))
    
# 打印所有图片的路径
for i in image_paths:
    # print(i)
    annotate(i)