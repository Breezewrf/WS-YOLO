"""
用于统计生成的伪标签中各个类别的样本数目
"""
from glob import glob
class_num_dict = {}
files = glob("/mnt/shared/wrf/yolov8/surgicalloc_v8_1/labels/*")
for file in files:
    print(file)
    with open(file) as f:
        for line in f:
            label, *_ = line.split()
            if label in class_num_dict.keys():
                class_num_dict[label] += 1
            else:
                class_num_dict[label] = 1
sorted_dict = sorted(class_num_dict.items(), key=lambda item: item[0])
print(sorted_dict)
