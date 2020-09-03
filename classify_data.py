from pathlib import Path
import csv
import os
from shutil import copyfile
import random


DATA_ROOT='/mnt/hdd1/home/joycenerd/industrial_defect_detection/aoi'

train_file=open(Path(DATA_ROOT).joinpath('train.csv'))
train_dir=Path(DATA_ROOT).joinpath('train_images')

defect_list=[]
normal=[]
defect_1=[]
defect_2=[]
defect_3=[]
defect_4=[]
defect_5=[]

reader=csv.reader(train_file)
next(reader)

print('start classification...')

for row in reader:
    name,defect=row[0],row[1]
    if not os.path.isdir(Path(train_dir).joinpath(defect)):
        os.mkdir(Path(train_dir).joinpath(defect))
    src=Path(train_dir).joinpath(name)
    dst=Path(train_dir).joinpath(defect,name)
    copyfile(src,dst)
    if defect=='0':
        normal.append(name)
    elif defect=='1':
        defect_1.append(name)
    elif defect=='2':
        defect_2.append(name)
    elif defect=='3':
        defect_3.append(name)
    elif defect=='4':
        defect_4.append(name)
    elif defect=='5':
        defect_5.append(name)

print('classification complete!')

defect_list.append(normal)
defect_list.append(defect_1)
defect_list.append(defect_2)
defect_list.append(defect_3)
defect_list.append(defect_4)
defect_list.append(defect_5)


print('start splitting train and valid...')

for idx, category in enumerate(defect_list):
    random.shuffle(category)
    for i in range(50):
        src=Path(train_dir).joinpath(str(idx),category[i])
        if not os.path.isdir(Path(train_dir).joinpath(str(idx),'train')):
            os.mkdir(Path(train_dir).joinpath(str(idx),'train'))
        dst=Path(train_dir).joinpath(str(idx),'train',category[i])
        copyfile(src,dst)
    for j in range(50,100):
        src=Path(train_dir).joinpath(str(idx),category[j])
        if not os.path.isdir(Path(train_dir).joinpath(str(idx),'valid')):
            os.mkdir(Path(train_dir).joinpath(str(idx),'valid'))
        dst=Path(train_dir).joinpath(str(idx),'valid',category[j])
        copyfile(src,dst)
    print(f'finist defect {idx} splitting!')

print('finish splitting train and valid!')





