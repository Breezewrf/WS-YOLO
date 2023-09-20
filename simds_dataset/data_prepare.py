"""
对multi-domain数据集进行划分，train_obj是面向器械类别的划分方式，train是面向part的划分方式
"""
import cv2
from glob import glob
from pathlib import Path
import shutil
datasets = ['exvivo_laparoscopic', 'exvivo_robotic', 'invivo_laparoscopic', 'invivo_robotic']
folders = []

train_dir = Path('./train_obj')
test_dir = Path('./test_obj')
train_img_dir = train_dir.joinpath('images')
train_mask_dir = train_dir.joinpath('masks')
test_img_dir = test_dir.joinpath('images')
test_mask_dir = test_dir.joinpath('masks')
dirs = [train_dir, test_dir, train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]
for dir in dirs:
    if not dir.exists():
        dir.mkdir()

for dataset in datasets:
    data_path = Path(dataset)
    sub_datasets = data_path.glob('*')
    for sub_dataset in sub_datasets:
        if '.tsv' not in str(sub_dataset):
            sub_dataset_path = sub_dataset
            folders.append(sub_dataset_path)

print(len(folders))

def get_data(folder_path=Path('exvivo_laparoscopic/01')):
    cnt = 0
    img_dir = folder_path.joinpath('images')
    mask_dir = folder_path.joinpath('masks')
    img_path = sorted(list(img_dir.glob('*.png')))
    mask_path = sorted(list(mask_dir.glob('*type_complex.png')))
    assert len(img_path) == len(mask_path), f"the number of images {len(img_path)} is not equal to the number of masks {len(mask_path)}"
   
    for idx, im in enumerate(img_path):
        if cnt % 5 == 0:  # test
            shutil.copy(list(img_path)[idx], test_img_dir.joinpath(str(folder_path).replace('/', '_')+'_'+(str(im).split('/')[-1])))
            shutil.copy(list(mask_path)[idx], test_mask_dir.joinpath(str(folder_path).replace('/', '_')+'_'+(str(im).split('/')[-1])))
            print(f"source: {list(img_path)[idx]}  {list(mask_path)[idx]}")
            print(f"destination: {test_img_dir.joinpath(str(folder_path).replace('/', '_')+'_'+(str(im).split('/')[-1]))}  {test_mask_dir.joinpath(str(folder_path).replace('/', '_')+'_'+(str(im).split('/')[-1]))} \n")
        else:
            shutil.copy(list(img_path)[idx], train_img_dir.joinpath(str(folder_path).replace('/', '_')+'_'+(str(im).split('/')[-1])))
            shutil.copy(list(mask_path)[idx], train_mask_dir.joinpath(str(folder_path).replace('/', '_')+'_'+(str(im).split('/')[-1])))
        cnt += 1

for folder in folders:
    get_data(folder_path=folder)