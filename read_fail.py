'''
read the failed data  (non-accurate detection) and save it to the folder
'''

from shutil import copyfile
import os


fail_file="/mnt/hdd1/home/joycenerd/industrial_defect_detection/aoi/Data/detectDefect/fail_data_name.txt"
image_dir="/mnt/hdd1/home/joycenerd/industrial_defect_detection/aoi/Data/valid"
target_dir="/mnt/hdd1/home/joycenerd/industrial_defect_detection/aoi/Data/fail"

os.makedirs(target_dir,exist_ok=True)

f=open(fail_file,'r')
line=f.readline()
while line:
    line=line.strip('\n')
    src=image_dir+'/'+line
    dst=target_dir+'/'+line
    copyfile(src,dst)
    line=f.readline()
