# industrial_defect_detection

## Introduction

Use deep learning to detect industrial defect. We want to not only classify the defect category but also detect the defect area of the industial defect.

## Method

First we classify the defect into 6 categories: 1 normal and 6 different defects: void, horizontal, vertical, edges, particle. Then we choose 50 images from each category as train and the other 50 as validation. We label the data using [VGG image annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html). Then we parse the label json file to COCO label format in order to send into [Detectron2](https://github.com/facebookresearch/detectron2) for training and validation.

## Result

<img src="./image/valid_00326_class1.png" style="width:100%" />
<span class="caption">Defect1: void</span>
<img src="./image/valid_00325_class2.png" style="width:100%" />
<span class="caption">Defect2: horizontal</span>
<img src="./image/valid_00384_class3.png" style="width:100%" />
<span class="caption">Defect3: vertical</span>
<img src="./image/valid_00770_class4.png" style="width:100%" />
<span class="caption">Defect4: edge</span>
 <img src="./image/valid_00081_class5.png" style="width:100%" />
<span class="caption">Defect5: particle</span>