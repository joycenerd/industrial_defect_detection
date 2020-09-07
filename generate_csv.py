'''
Generate CSV file for train and validation data
'''

import pandas as pd
import os
from pathlib import Path


DATAROOT='/mnt/hdd1/home/joycenerd/industrial_defect_detection/aoi/Data'

train_image_name=[]
train_defect_type=[]

valid_image_name=[]
valid_defect_type=[]

for defect_dir in os.listdir(DATAROOT):
    if os.path.isfile(Path(DATAROOT).joinpath(defect_dir)):
        continue
    
    train_dir=Path(DATAROOT).joinpath(defect_dir,'train')
    for image in os.listdir(train_dir):
        train_image_name.append(image)
        train_defect_type.append(defect_dir)
    
    valid_dir=Path(DATAROOT).joinpath(defect_dir,'valid')
    for image in os.listdir(valid_dir):
        valid_image_name.append(image)
        valid_defect_type.append(defect_dir)

data = {
    'Image Name': train_image_name,
    'Defect Type':train_defect_type
}

df=pd.DataFrame(data,columns=['Image Name','Defect Type'])
df.to_csv(Path(DATAROOT).joinpath('train.csv'),index=False)

data={
    'Image Name': valid_image_name,
    'Defect Type': valid_defect_type
}

df=pd.DataFrame(data,columns=['Image Name','Defect Type'])
df.to_csv(Path(DATAROOT).joinpath('valid.csv'),index=False)