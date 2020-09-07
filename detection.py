'''
Detectron2 from facebook AI researchs
'''


import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import json
import cv2
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


setup_logger()


# make coco format dataset
def get_dicts(img_dir,annot_json):
    json_file=os.path.join(img_dir,annot_json)
    print(json_file)
    with open(json_file) as f:
        imgs_anns=json.load(f)

    dataset_dicts=[]
    for idx,v in enumerate(imgs_anns.values()):
        record={}

        filename=os.path.join(img_dir,v['filename'])
        
        if annot_json=='valid.json':
            #print(filename)
            words=filename.split('_')
            new_word='c'+words[-1][1:]
            filename='_'.join(words[:-1])
            filename+='_'+new_word
            #print(filename)

        height,width=cv2.imread(filename).shape[:2]

        record['file_name']=filename
        record['image_id']=idx
        record['height']=height
        record['width']=width

        annos=v['regions']
        objs=[]
        for anno in annos:
            anno_1=anno['shape_attributes']
            anno_2=anno['region_attributes']
            px=anno_1['all_points_x']
            py=anno_1['all_points_y']
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)]
            poly=[p for x in poly for p in x]
            obj={
                'bbox':[np.mean(px),np.mean(py),np.max(px),np.max(py)],
                'bbox_mode':BoxMode.XYXY_ABS,
                'segmentation':[poly],
                'category_id':int(anno_2['defect']),
                'iscrowd':0
            }
            objs.append(obj)
        #print(f'{idx+1}: {filename}')
        record['annotations']=objs
        dataset_dicts.append(record)
    return dataset_dicts

# registering data
path='/mnt/hdd1/home/joycenerd/industrial_defect_detection/aoi/Data'
for d in ['train','valid']:
    DatasetCatalog.register('defect_'+d,lambda d=d:get_dicts(path+'/'+d,d+'.json'))
    MetadataCatalog.get('defect_'+d).set(thing_classes=['1','2','3','4','5'])

# Settings for training the model
cfg=get_cfg()
cfg.OUTPUT_DIR=path+'/'+'detectDefect'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # get pre-trained model

cfg.DATASETS.TRAIN=('defect_train',)
cfg.DATASETS.TEST=()
cfg.DATALOADER.NUM_WORKERS=2
cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH=2 # batch size
cfg.SOLVER.BASE_LR=0.00025
cfg.SOLVER.MAX_ITER=3600
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128 # number of proposals to sample for training
cfg.MODEL.ROI_HEADS.NUM_CLASSES=5 # have 5 class
cfg.MODEL.DEVICE='cuda:1'

# training
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
trainer=DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.build_writers()
trainer.train()


cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRSH_TEST=0.7 # testing threshold
cfg.DATASETS.TEST=("defect_valid")
predictor=DefaultPredictor(cfg)

# validation
dataset_dicts=get_dicts(path+'/valid','valid.json')
predict_dir=os.path.join(cfg.OUTPUT_DIR,'predict_images')
os.makedirs(predict_dir,exist_ok=True)
for d in dataset_dicts:
    im=cv2.imread(d['file_name'])
    outputs=predictor(im)
    v=Visualizer(im[:,:,::-1],metadata=MetadataCatalog.get('defect_valid'),scale=0.8,instance_mode=ColorMode.IMAGE_BW) # remove the color for unsegmented pixels
    v=v.draw_instance_predictions(outputs['instances'].to('cpu'))
    filename=d['file_name'].split('/')
    filepath=str(os.path.join(cfg.OUTPUT_DIR,'predict_images',filename[-1]))
    print(filepath)
    cv2.imwrite(filepath,v.get_image()[:,:,::-1])

# evaluate model
inference_dir=os.path.join(cfg.OUTPUT_DIR,'inference')
os.makedirs(inference_dir,exist_ok=True)
evaluator=COCOEvaluator('defect_valid',cfg,False,output_dir=inference_dir)
val_loader=build_detection_test_loader(cfg,'defect_valid')
print(inference_on_dataset(trainer.model,val_loader,evaluator))






