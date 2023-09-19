# instructions:
./simds_dataset/data_prepare.py 将数据集划分为train和test
./simds_dataset/get_annotation.py 将mask转换为txt label
# train Det_parts:
yolo detect train data=simd_det.yaml model=yolox.pt imsz=640 epochs=1000 

# use ./sample_util.py to sample video or images

# predict parts:
yolo detect predict model=runs/detect/train6/weights/best.pt source=../surgtooloc2022_dataset/sampled_data/ save=False save_txt=True project=regenerate_round1 name=bbox1

# initial match: 取消注释./yolo/engine/predictor.py中breezewrf部分的代码
# pseudo_label_path = "/mnt/shared/wrf/yolov8/regenerate_round1/bbox2/labels"
yolo detect predict model=runs/detect/train6/weights/best.pt source=../surgtooloc2022_dataset/sampled_data/ save=False save_txt=True
# 注释./yolo/engine/predictor.py中breezewrf部分的代码

# train Det_tools:
yolo detect train data=simd_det.yaml model=yolox.pt imsz=640 epochs=1000

# predict tools:
yolo detect predict model=runs/detect/train7/weights/best.pt source=../surgtooloc2022_dataset/sampled_data/ save=False save_txt=True project=regenerate_round2 name=bbox2

# multi-round match
python yolov8/regenerate_round1/match_util.py

# return to train Det_tools, bbox1 is copied from regenerated_round1

# some test demo: yolov8/test

# summary the classes distribution
python ./summary.py

---------------------------------------------------------
# transfer model to submit
# in Ubuntu desktop
scp server:/mnt/shared/wrf/yolov8/runs/detect/train25/weights/best.pt .
# change the pth file name in process.py and Dockerfile
sudo sh build.sh
sudo sh test.sh # never mind the Error of tmp not exist
sudo sh export.sh
# submit