import os

dir = '/mnt/shared/wrf/yolov8/dataset_manually/labels'
mapper = {'0': '6', '1': '11', '2': '4', '3': '5', '4': '8', '5': '9', '6': '12', '7': '3'}
for filename in os.listdir(dir):
    if filename.endswith('.txt'):
        with open(os.path.join(dir, filename)) as f:
            lines = f.readlines()
        with open(os.path.join(dir, filename), 'w') as f:
           for line in lines:
                line_list = line.split(' ')
                line_list[0] = mapper[line_list[0]]
                f.write(' '.join(line_list))
                # print()