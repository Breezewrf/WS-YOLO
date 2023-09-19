import numpy as np
import os
from glob import glob
from pathlib import Path
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.metrics import box_iou
import csv
import shutil
import natsort

label_map = {0: "needle driver",
                    1: "monopolar curved scissors",
                    2: "force bipolar",
                    3: "clip applier",
                    4: "tip-up fenestrated grasper",
                    5: "cadiere forceps",
                    6: "bipolar forceps",
                    7: "vessel sealer",
                    8: "suction irrigator",
                    9: "bipolar dissector",
                    10: "prograsp forceps",
                    11: "stapler",
                    12: "permanent cautery hook/spatula",
                    13: "grasping retractor"
                    }
special_classes = ["monopolar curved scissors", "tip-up fenestrated grasper", "suction irrigator", "stapler", "grasping retractor"]
clip_instruments = {}
with open('/mnt/shared/wrf/surgtoolloc2022_dataset/_release/training_data/labels.csv') as f:
    reader = csv.reader(f)
    next(reader) # 跳过标题行
    for row in reader:
        clip_id, instruments = row[0], row[2]
        clip_instruments['clip_{}'.format(clip_id.zfill(6))] = instruments

def compute_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

def get_file_list(bbox1_dir, output_dir):
    bbox1_files = glob(bbox1_dir+'/*')
    bbox1_files = natsort.natsorted(bbox1_files)
    cnt = 0
    for file in bbox1_files:
        clip_id = file.split('/')[-1].split('.')[0][:11]
        instruments = clip_instruments[clip_id].strip('[]').split(', ')
        instruments = [i.strip(' ') for i in instruments if i != 'nan']
        assert len(instruments) == 3, print(f'please check {instruments}')
        bbox2_file = file.replace('bbox1', 'bbox2').replace("regenerate_round3", "regenerate_round1")
        bbox1_list = []
        bbox2_list = []
        label1_list = []
        label2_list = []
        next_flag = False
        # print("here")
        with open(file, 'r') as f:
            for line in f.readlines():
                fields = line.strip().split()
                label = int(fields[0])
                if label_map[label] not in instruments:  # bbox1类别检测错误
                    next_flag = True
                    print(f"{label} not in {instruments} in video {file}")
                    break
                label1_list.append(label)
                x, y, w, h = map(float, fields[1:])
                bbox1_list.append(xywh2xyxy(np.array([x, y, w, h])))
                # print(line)
        if next_flag:
            continue
        
        try:
            with open(bbox2_file, 'r') as f2:
                for line in f2.readlines():
                    fields = line.strip().split()
                    label = int(fields[0])
                    label2_list.append(label)
                    x, y, w, h = map(float, fields[1:])
                    bbox2_list.append(xywh2xyxy(np.array([x, y, w, h])))
        except:
            print(f"{bbox2_file} path error")
            continue
        # print(bbox1_list, '\n', bbox2_list)
        for i in range(len(bbox1_list)):  # bbox1中的每个bbox都要有匹配到的bbox2
            iou_list = []
            for j in range(len(bbox2_list)):
                bbox1 = [bbox1_list[i][0], bbox1_list[i][1],
                        bbox1_list[i][2], bbox1_list[i][3]]
                bbox2 = [bbox2_list[j][0], bbox2_list[j][1],
                        bbox2_list[j][2], bbox2_list[j][3]]
                iou = compute_iou(bbox1, bbox2)
                iou_list.append(iou)
                # print(f"bbox1[{i}] and bbox2[{j}]: {iou:.4f}")
            if label_map[label1_list[i]] in special_classes and label2_list[np.array(iou_list).argmax()] != 2:
                next_flag = True
                print(f"label1: {label_map[label1_list[i]]} in special list {special_classes} and {label2_list[np.array(iou_list).argmax()]} != 2")
                break
            elif label_map[label1_list[i]] not in special_classes and label2_list[np.array(iou_list).argmax()] != 1:
                next_flag = True
                print(f"label1: {label_map[label1_list[i]]} not in special list {special_classes} and {label2_list[np.array(iou_list).argmax()]} != 1")
                break
        if not next_flag:
            image_path = file.replace('bbox1/labels', 'images').replace('txt', 'jpg').replace("regenerate_round3", "regenerate_round1")
            if os.path.exists(image_path):
                print(f"{image_path} founded!")
                shutil.copy(image_path, output_dir + "/images/" + image_path.split('/')[-1])
                shutil.copy(file, output_dir + "/labels/" + file.split('/')[-1])
            else:
                print(f"{image_path} is not founded!")
        # cnt += 1
        # if cnt == 10:
        #     break

    # print(bbox1_files[:5])



get_file_list('./regenerate_round3/bbox1/labels', output_dir='/mnt/shared/wrf/yolov8/surgicalloc_v7/')



