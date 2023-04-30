# Dogs Pose Recognition

[![Roboflow Dataset](https://raw.githubusercontent.com/roboflow/notebooks/main/assets/badges/roboflow-dataset.svg)](https://universe.roboflow.com/emgs/dogs-pose-recognition)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/emgs/dogs-pose-recognition)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter_notebook-%23FF7A01.svg?logo=jupyter&logoColor=white)](notebooks/Dogs_Pose_Recognition.ipynb)

<img src="./assets/banner.jpg" alt="Dog laying down" width="300" height="300">

## About the project

The main goal of this project is to utilize object detection to identify my dogs in three different positions: squatting, sitting, and lying down. As I don't have enough photos of them, I have utilized various data sources to create a comprehensive dataset:

- **Squatting**: I obtained the images from a public Instagram account called ["dogspoopinginprettyplaces"](https://www.instagram.com/dogspoopinginprettyplaces/?hl=en) using the [instaloader](https://instaloader.github.io/) library.
- **Sitting**: I manually selected similar dog images from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) that were in a sitting position.
- **Lying Down**: While most of the images are from my own dogs, I also added a few more from the internet using the ["bing_image_downloader"](https://pypi.org/project/bing-image-downloader/) library.


## Training and Performance

|**Project Type**|**Model**|**Library**|**Performance**
|--|--|--|--|
|Object Detection|YOLOv5|PyTorch|?|

### Reports

[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://api.wandb.ai/links/emgs/hsbkxumh)

### Logs
<details>
  <summary>Click to expand!</summary>
  
  ```text
    wandb: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
wandb: Currently logged in as: emgs. Use `wandb login --relogin` to force relogin
train: weights=yolov5s.pt, cfg=, data=/content/yolov5/yolov5/Dogs-Pose-Recognition-13/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=4000, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=true, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v7.0-158-g8211a03 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Tracking run with wandb version 0.15.0
wandb: Run data is saved locally in /content/yolov5/yolov5/wandb/run-20230430_183631-19jirz9b
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run rich-haze-1
wandb: ⭐️ View project at https://wandb.ai/emgs/YOLOv5
wandb: 🚀 View run at https://wandb.ai/emgs/YOLOv5/runs/19jirz9b
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 26.0MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 212MB/s]

Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     21576  models.yolo.Detect                      [3, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning /content/yolov5/yolov5/Dogs-Pose-Recognition-13/train/labels... 508 images, 0 backgrounds, 0 corrupt: 100% 508/508 [00:00<00:00, 966.55it/s] 
train: New cache created: /content/yolov5/yolov5/Dogs-Pose-Recognition-13/train/labels.cache
train: Caching images (0.6GB true): 100% 508/508 [00:03<00:00, 137.39it/s]
val: Scanning /content/yolov5/yolov5/Dogs-Pose-Recognition-13/valid/labels... 129 images, 0 backgrounds, 0 corrupt: 100% 129/129 [00:00<00:00, 559.05it/s]
val: New cache created: /content/yolov5/yolov5/Dogs-Pose-Recognition-13/valid/labels.cache
val: Caching images (0.1GB true): 100% 129/129 [00:02<00:00, 61.84it/s] 

AutoAnchor: 3.20 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/exp/labels.jpg... 
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 4000 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     0/3999      3.49G    0.09166    0.02936    0.03698         31        640: 100% 32/32 [00:13<00:00,  2.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:02<00:00,  1.92it/s]
                   all        129        138     0.0667      0.348     0.0946     0.0354

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     1/3999       4.3G    0.06613      0.025    0.02991         29        640: 100% 32/32 [00:08<00:00,  3.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.10it/s]
                   all        129        138      0.321      0.504      0.286      0.132

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     2/3999       4.3G    0.05877     0.0216    0.02589         33        640: 100% 32/32 [00:06<00:00,  4.96it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.39it/s]
                   all        129        138      0.238       0.33      0.252      0.102

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     3/3999       4.3G    0.06173    0.01964    0.02176         33        640: 100% 32/32 [00:07<00:00,  4.15it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.74it/s]
                   all        129        138       0.39      0.288      0.289      0.124

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     4/3999       4.3G    0.05339    0.01837    0.02358         27        640: 100% 32/32 [00:07<00:00,  4.57it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.93it/s]
                   all        129        138      0.374      0.343      0.267     0.0945

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     5/3999       4.3G     0.0488    0.01892    0.02252         25        640: 100% 32/32 [00:07<00:00,  4.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138       0.28      0.444      0.313      0.136

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     6/3999       4.3G    0.04714    0.01843    0.01973         32        640: 100% 32/32 [00:07<00:00,  4.03it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.45it/s]
                   all        129        138      0.328       0.39      0.311      0.138

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     7/3999       4.3G    0.04538    0.01792    0.02032         32        640: 100% 32/32 [00:06<00:00,  4.92it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.59it/s]
                   all        129        138      0.431      0.467      0.412      0.176

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     8/3999       4.3G     0.0444    0.01685    0.01961         29        640: 100% 32/32 [00:08<00:00,  3.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.74it/s]
                   all        129        138      0.365      0.414      0.322      0.125

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     9/3999       4.3G    0.04453    0.01613    0.01702         21        640: 100% 32/32 [00:06<00:00,  4.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.79it/s]
                   all        129        138      0.263      0.261      0.209     0.0765

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    10/3999       4.3G    0.04245    0.01665    0.02036         36        640: 100% 32/32 [00:07<00:00,  4.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.70it/s]
                   all        129        138      0.201      0.347      0.243      0.107

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    11/3999       4.3G    0.04188    0.01607     0.0166         19        640: 100% 32/32 [00:07<00:00,  4.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.99it/s]
                   all        129        138      0.365      0.449      0.363       0.16

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    12/3999       4.3G    0.03962     0.0161    0.01692         31        640: 100% 32/32 [00:07<00:00,  4.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.63it/s]
                   all        129        138      0.441      0.241      0.284      0.116

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    13/3999       4.3G    0.03895     0.0166    0.01581         27        640: 100% 32/32 [00:07<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.88it/s]
                   all        129        138      0.307      0.414      0.368      0.153

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    14/3999       4.3G    0.04046    0.01576    0.01705         26        640: 100% 32/32 [00:06<00:00,  4.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138       0.31      0.399      0.376      0.182

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    15/3999       4.3G    0.03833    0.01555    0.01571         27        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.27it/s]
                   all        129        138      0.488      0.474      0.447       0.19

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    16/3999       4.3G    0.03935     0.0154     0.0157         26        640: 100% 32/32 [00:06<00:00,  4.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.52it/s]
                   all        129        138      0.437      0.486      0.409      0.168

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    17/3999       4.3G    0.03683    0.01453    0.01534         31        640: 100% 32/32 [00:08<00:00,  3.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.71it/s]
                   all        129        138      0.311      0.357      0.294      0.108

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    18/3999       4.3G    0.03773    0.01518     0.0149         21        640: 100% 32/32 [00:06<00:00,  4.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.24it/s]
                   all        129        138      0.262      0.363      0.299       0.13

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    19/3999       4.3G    0.03719    0.01652    0.01588         24        640: 100% 32/32 [00:07<00:00,  4.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.70it/s]
                   all        129        138      0.276      0.414      0.295      0.095

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    20/3999       4.3G    0.03583    0.01567    0.01587         31        640: 100% 32/32 [00:07<00:00,  4.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.80it/s]
                   all        129        138      0.512      0.486      0.458      0.208

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    21/3999       4.3G    0.03501    0.01515    0.01322         32        640: 100% 32/32 [00:06<00:00,  4.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.73it/s]
                   all        129        138       0.38      0.513      0.404      0.164

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    22/3999       4.3G    0.03487    0.01539    0.01366         30        640: 100% 32/32 [00:08<00:00,  3.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:02<00:00,  2.29it/s]
                   all        129        138      0.334      0.402       0.35      0.149

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    23/3999       4.3G    0.03548    0.01534    0.01433         26        640: 100% 32/32 [00:06<00:00,  4.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.71it/s]
                   all        129        138      0.528      0.559      0.503      0.201

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    24/3999       4.3G    0.03374    0.01436    0.01521         32        640: 100% 32/32 [00:08<00:00,  3.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.82it/s]
                   all        129        138      0.493      0.512      0.501      0.253

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    25/3999       4.3G    0.03362    0.01377     0.0127         35        640: 100% 32/32 [00:06<00:00,  4.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.513      0.562      0.474      0.214

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    26/3999       4.3G    0.03371    0.01373    0.01336         19        640: 100% 32/32 [00:08<00:00,  3.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.51it/s]
                   all        129        138      0.568      0.483      0.441      0.179

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    27/3999       4.3G    0.03201    0.01426    0.01245         30        640: 100% 32/32 [00:06<00:00,  4.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.53it/s]
                   all        129        138      0.502      0.526      0.422      0.163

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    28/3999       4.3G    0.03274    0.01403    0.01212         21        640: 100% 32/32 [00:07<00:00,  4.02it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.77it/s]
                   all        129        138      0.397      0.447      0.403      0.179

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    29/3999       4.3G    0.03227    0.01411    0.01177         29        640: 100% 32/32 [00:07<00:00,  4.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.26it/s]
                   all        129        138      0.412      0.284      0.277      0.129

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    30/3999       4.3G     0.0312    0.01378    0.01308         33        640: 100% 32/32 [00:07<00:00,  4.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.81it/s]
                   all        129        138      0.378      0.406      0.373      0.172

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    31/3999       4.3G    0.03297    0.01408    0.01266         31        640: 100% 32/32 [00:07<00:00,  4.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.62it/s]
                   all        129        138      0.481      0.461      0.421      0.201

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    32/3999       4.3G    0.03025    0.01376    0.01305         28        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.71it/s]
                   all        129        138      0.483      0.603      0.515      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    33/3999       4.3G    0.03276    0.01368    0.01198         24        640: 100% 32/32 [00:08<00:00,  3.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.38it/s]
                   all        129        138      0.474      0.561      0.469      0.195

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    34/3999       4.3G    0.02978    0.01285    0.01132         26        640: 100% 32/32 [00:06<00:00,  4.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.66it/s]
                   all        129        138      0.455      0.452       0.42      0.191

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    35/3999       4.3G     0.0301    0.01324    0.01326         32        640: 100% 32/32 [00:08<00:00,  3.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.79it/s]
                   all        129        138      0.476      0.466      0.388      0.153

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    36/3999       4.3G    0.03117    0.01409    0.01234         26        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.43it/s]
                   all        129        138      0.428      0.436      0.413      0.189

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    37/3999       4.3G    0.02993    0.01297    0.01198         30        640: 100% 32/32 [00:07<00:00,  4.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.78it/s]
                   all        129        138      0.516      0.473      0.414      0.197

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    38/3999       4.3G    0.03093    0.01388    0.01268         28        640: 100% 32/32 [00:06<00:00,  4.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.01it/s]
                   all        129        138      0.508      0.487      0.468      0.202

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    39/3999       4.3G    0.03005    0.01273    0.01069         28        640: 100% 32/32 [00:07<00:00,  4.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.71it/s]
                   all        129        138      0.519      0.514      0.514      0.232

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    40/3999       4.3G    0.03043    0.01289    0.01207         30        640: 100% 32/32 [00:07<00:00,  4.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.67it/s]
                   all        129        138      0.394      0.412      0.371      0.171

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    41/3999       4.3G    0.02904    0.01265     0.0101         23        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.68it/s]
                   all        129        138      0.488      0.537      0.496      0.215

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    42/3999       4.3G    0.03159    0.01247      0.011         30        640: 100% 32/32 [00:08<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.49it/s]
                   all        129        138      0.469      0.493      0.429      0.182

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    43/3999       4.3G    0.02809     0.0135    0.01013         35        640: 100% 32/32 [00:06<00:00,  4.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.85it/s]
                   all        129        138      0.334      0.459      0.403      0.178

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    44/3999       4.3G     0.0273    0.01269   0.009368         24        640: 100% 32/32 [00:08<00:00,  3.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.427       0.51      0.463        0.2

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    45/3999       4.3G    0.02966    0.01225    0.01064         28        640: 100% 32/32 [00:06<00:00,  4.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.41it/s]
                   all        129        138      0.513       0.59      0.473      0.209

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    46/3999       4.3G    0.02891    0.01188   0.009781         26        640: 100% 32/32 [00:09<00:00,  3.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.42it/s]
                   all        129        138      0.512      0.544      0.502      0.234

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    47/3999       4.3G    0.02791     0.0115     0.0104         30        640: 100% 32/32 [00:07<00:00,  4.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.66it/s]
                   all        129        138      0.384      0.386      0.339      0.157

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    48/3999       4.3G    0.02931    0.01254   0.009088         30        640: 100% 32/32 [00:06<00:00,  4.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.75it/s]
                   all        129        138      0.494      0.452      0.421      0.196

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    49/3999       4.3G    0.02752     0.0122   0.009144         20        640: 100% 32/32 [00:08<00:00,  3.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.62it/s]
                   all        129        138       0.51      0.585      0.513      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    50/3999       4.3G    0.02788    0.01225   0.009017         28        640: 100% 32/32 [00:06<00:00,  4.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.68it/s]
                   all        129        138      0.547      0.537      0.495      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    51/3999       4.3G    0.02653    0.01192   0.009865         22        640: 100% 32/32 [00:08<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.544      0.526      0.496      0.229

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    52/3999       4.3G    0.02659     0.0117   0.009276         27        640: 100% 32/32 [00:06<00:00,  4.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.19it/s]
                   all        129        138      0.438      0.504      0.412      0.163

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    53/3999       4.3G    0.02584     0.0119   0.009462         38        640: 100% 32/32 [00:07<00:00,  4.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.73it/s]
                   all        129        138      0.407      0.439      0.397      0.168

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    54/3999       4.3G     0.0262    0.01219   0.009558         22        640: 100% 32/32 [00:07<00:00,  4.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.82it/s]
                   all        129        138      0.487      0.471      0.455      0.215

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    55/3999       4.3G    0.02506    0.01223   0.008708         22        640: 100% 32/32 [00:06<00:00,  4.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.75it/s]
                   all        129        138       0.53      0.333      0.385      0.201

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    56/3999       4.3G    0.02597    0.01156   0.007559         27        640: 100% 32/32 [00:08<00:00,  3.94it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.39it/s]
                   all        129        138      0.516      0.495      0.439      0.181

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    57/3999       4.3G    0.02673     0.0114   0.009895         27        640: 100% 32/32 [00:06<00:00,  4.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.85it/s]
                   all        129        138      0.535      0.514      0.443      0.195

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    58/3999       4.3G    0.02467    0.01167    0.00947         28        640: 100% 32/32 [00:08<00:00,  3.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.447      0.438      0.447      0.198

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    59/3999       4.3G    0.02579    0.01189   0.009699         33        640: 100% 32/32 [00:06<00:00,  4.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.80it/s]
                   all        129        138      0.487      0.405      0.457      0.217

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    60/3999       4.3G    0.02598    0.01219   0.008744         30        640: 100% 32/32 [00:07<00:00,  4.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.70it/s]
                   all        129        138       0.41      0.478      0.413      0.214

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    61/3999       4.3G    0.02615    0.01208    0.00944         35        640: 100% 32/32 [00:06<00:00,  4.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.37it/s]
                   all        129        138      0.406      0.353      0.358      0.174

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    62/3999       4.3G    0.02705     0.0123   0.008714         34        640: 100% 32/32 [00:07<00:00,  4.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.68it/s]
                   all        129        138      0.597      0.464      0.493      0.229

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    63/3999       4.3G     0.0251     0.0115    0.01007         25        640: 100% 32/32 [00:07<00:00,  4.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.87it/s]
                   all        129        138      0.422      0.506      0.429       0.19

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    64/3999       4.3G     0.0251    0.01139   0.009266         25        640: 100% 32/32 [00:06<00:00,  4.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.64it/s]
                   all        129        138      0.506      0.558      0.486      0.227

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    65/3999       4.3G    0.02529    0.01058   0.008414         27        640: 100% 32/32 [00:08<00:00,  3.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.69it/s]
                   all        129        138      0.482      0.563      0.485      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    66/3999       4.3G    0.02533    0.01107   0.008647         27        640: 100% 32/32 [00:06<00:00,  4.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.88it/s]
                   all        129        138      0.522      0.476      0.441      0.203

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    67/3999       4.3G    0.02405    0.01163   0.008558         34        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.59it/s]
                   all        129        138      0.481      0.499      0.425      0.195

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    68/3999       4.3G     0.0244    0.01132   0.008332         32        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.80it/s]
                   all        129        138      0.491      0.442      0.436       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    69/3999       4.3G     0.0252    0.01072   0.009659         34        640: 100% 32/32 [00:08<00:00,  3.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.41it/s]
                   all        129        138      0.524      0.522       0.44      0.204

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    70/3999       4.3G     0.0235    0.01117    0.00829         32        640: 100% 32/32 [00:09<00:00,  3.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.00it/s]
                   all        129        138      0.513      0.473      0.442      0.201

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    71/3999       4.3G     0.0238    0.01118   0.007668         26        640: 100% 32/32 [00:06<00:00,  4.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.79it/s]
                   all        129        138      0.459      0.442      0.396      0.172

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    72/3999       4.3G    0.02406    0.01142   0.007921         24        640: 100% 32/32 [00:08<00:00,  3.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.502      0.364      0.413      0.171

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    73/3999       4.3G    0.02423    0.01118   0.008204         24        640: 100% 32/32 [00:06<00:00,  4.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.32it/s]
                   all        129        138      0.444      0.414      0.386      0.184

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    74/3999       4.3G    0.02373    0.01039   0.008609         29        640: 100% 32/32 [00:08<00:00,  3.92it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.82it/s]
                   all        129        138      0.533      0.408      0.436      0.211

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    75/3999       4.3G     0.0258    0.01061   0.008657         28        640: 100% 32/32 [00:06<00:00,  4.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.09it/s]
                   all        129        138      0.552      0.546      0.496      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    76/3999       4.3G    0.02587     0.0111    0.00976         35        640: 100% 32/32 [00:07<00:00,  4.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.81it/s]
                   all        129        138      0.483      0.562      0.454      0.214

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    77/3999       4.3G    0.02513    0.01141   0.007843         24        640: 100% 32/32 [00:07<00:00,  4.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.51it/s]
                   all        129        138      0.489      0.557      0.456      0.224

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    78/3999       4.3G     0.0255    0.01097   0.007265         24        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.82it/s]
                   all        129        138      0.423      0.467      0.356      0.158

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    79/3999       4.3G    0.02631    0.01116   0.007679         31        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.99it/s]
                   all        129        138       0.36        0.4      0.305      0.129

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    80/3999       4.3G    0.02369    0.01087   0.008204         29        640: 100% 32/32 [00:06<00:00,  4.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.321       0.41       0.35      0.143

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    81/3999       4.3G    0.02361    0.01035   0.007632         26        640: 100% 32/32 [00:08<00:00,  3.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.81it/s]
                   all        129        138      0.408      0.405      0.399      0.197

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    82/3999       4.3G    0.02429    0.01106   0.008373         28        640: 100% 32/32 [00:06<00:00,  4.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.47it/s]
                   all        129        138      0.442      0.398      0.402      0.184

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    83/3999       4.3G    0.02378    0.01112   0.008162         36        640: 100% 32/32 [00:07<00:00,  4.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.77it/s]
                   all        129        138      0.495      0.504      0.447      0.194

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    84/3999       4.3G    0.02363    0.01094   0.009279         32        640: 100% 32/32 [00:07<00:00,  4.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.95it/s]
                   all        129        138       0.51      0.413      0.403      0.175

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    85/3999       4.3G    0.02277     0.0111   0.009523         20        640: 100% 32/32 [00:07<00:00,  4.45it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.51it/s]
                   all        129        138      0.509       0.52      0.447      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    86/3999       4.3G    0.02478    0.01053   0.009425         31        640: 100% 32/32 [00:07<00:00,  4.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.77it/s]
                   all        129        138      0.465      0.521      0.457      0.229

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    87/3999       4.3G    0.02407    0.01111   0.007633         30        640: 100% 32/32 [00:06<00:00,  4.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.53it/s]
                   all        129        138      0.522      0.515      0.472      0.221

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    88/3999       4.3G    0.02328    0.01074   0.007192         25        640: 100% 32/32 [00:08<00:00,  3.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.18it/s]
                   all        129        138      0.532      0.494      0.479      0.241

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    89/3999       4.3G    0.02385    0.01075   0.007074         26        640: 100% 32/32 [00:06<00:00,  4.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.74it/s]
                   all        129        138      0.527      0.435      0.462      0.218

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    90/3999       4.3G    0.02445    0.01077   0.007608         37        640: 100% 32/32 [00:08<00:00,  3.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.26it/s]
                   all        129        138       0.52      0.503      0.484      0.246

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    91/3999       4.3G     0.0241    0.01019   0.006296         22        640: 100% 32/32 [00:07<00:00,  4.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.99it/s]
                   all        129        138      0.527      0.538      0.481      0.223

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    92/3999       4.3G    0.02437    0.01082   0.007255         25        640: 100% 32/32 [00:07<00:00,  4.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.75it/s]
                   all        129        138      0.532      0.537      0.446      0.206

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    93/3999       4.3G    0.02338   0.009994   0.007511         25        640: 100% 32/32 [00:08<00:00,  3.95it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:02<00:00,  2.26it/s]
                   all        129        138       0.53      0.542      0.485      0.224

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    94/3999       4.3G    0.02326    0.01039   0.006273         39        640: 100% 32/32 [00:07<00:00,  4.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.73it/s]
                   all        129        138      0.605      0.467      0.492      0.242

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    95/3999       4.3G    0.02299    0.01053   0.006752         28        640: 100% 32/32 [00:07<00:00,  4.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.72it/s]
                   all        129        138      0.608      0.421      0.464      0.223

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    96/3999       4.3G    0.02169    0.01002   0.006551         26        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.89it/s]
                   all        129        138      0.457      0.408      0.441      0.227

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    97/3999       4.3G    0.02243    0.01046   0.006514         32        640: 100% 32/32 [00:08<00:00,  3.95it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.66it/s]
                   all        129        138      0.484       0.42      0.345      0.166

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    98/3999       4.3G    0.02194     0.0106   0.007987         26        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.85it/s]
                   all        129        138      0.525      0.414      0.403      0.195

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    99/3999       4.3G    0.02445    0.01069   0.009055         23        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.47it/s]
                   all        129        138      0.474      0.488       0.44      0.202

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   100/3999       4.3G    0.02236   0.009875   0.007674         30        640: 100% 32/32 [00:06<00:00,  4.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.43it/s]
                   all        129        138      0.577      0.546      0.526      0.234

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   101/3999       4.3G    0.02191    0.01071   0.006945         28        640: 100% 32/32 [00:08<00:00,  3.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.79it/s]
                   all        129        138        0.5      0.538      0.496      0.239

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   102/3999       4.3G    0.02247    0.01081   0.006734         30        640: 100% 32/32 [00:06<00:00,  4.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.28it/s]
                   all        129        138      0.478      0.554       0.47      0.212

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   103/3999       4.3G    0.02132   0.009872   0.006334         31        640: 100% 32/32 [00:07<00:00,  4.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.549       0.58      0.506      0.239

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   104/3999       4.3G    0.02221   0.009835   0.006353         25        640: 100% 32/32 [00:07<00:00,  4.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.66it/s]
                   all        129        138      0.561      0.532      0.488      0.219

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   105/3999       4.3G    0.02211    0.00945   0.005991         29        640: 100% 32/32 [00:06<00:00,  4.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.564      0.479      0.513      0.249

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   106/3999       4.3G     0.0216   0.009229   0.005306         30        640: 100% 32/32 [00:08<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.38it/s]
                   all        129        138      0.552      0.555      0.527      0.241

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   107/3999       4.3G     0.0224   0.009917   0.007161         22        640: 100% 32/32 [00:06<00:00,  4.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.71it/s]
                   all        129        138      0.548      0.518      0.502       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   108/3999       4.3G    0.02092   0.009639    0.00611         28        640: 100% 32/32 [00:08<00:00,  3.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.62it/s]
                   all        129        138      0.476      0.557      0.525      0.253

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   109/3999       4.3G    0.02249   0.009537   0.006649         27        640: 100% 32/32 [00:06<00:00,  4.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.35it/s]
                   all        129        138       0.54       0.54      0.511       0.24

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   110/3999       4.3G    0.02143    0.01046   0.006627         23        640: 100% 32/32 [00:07<00:00,  4.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.477      0.598      0.507      0.254

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   111/3999       4.3G    0.02269   0.009918    0.00614         29        640: 100% 32/32 [00:07<00:00,  4.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.79it/s]
                   all        129        138       0.53      0.544       0.51      0.228

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   112/3999       4.3G     0.0216   0.009395   0.006034         30        640: 100% 32/32 [00:06<00:00,  4.62it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.79it/s]
                   all        129        138      0.513      0.585      0.482      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   113/3999       4.3G    0.02289    0.01012   0.007349         26        640: 100% 32/32 [00:07<00:00,  4.06it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.14it/s]
                   all        129        138      0.524      0.481      0.461      0.206

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   114/3999       4.3G     0.0217    0.01007   0.006235         28        640: 100% 32/32 [00:06<00:00,  4.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.84it/s]
                   all        129        138      0.565       0.46      0.446      0.218

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   115/3999       4.3G     0.0215   0.009501   0.008266         29        640: 100% 32/32 [00:08<00:00,  3.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.51it/s]
                   all        129        138      0.548      0.541      0.477      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   116/3999       4.3G    0.02196    0.01045   0.007738         28        640: 100% 32/32 [00:06<00:00,  4.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.96it/s]
                   all        129        138      0.511      0.499      0.466      0.223

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   117/3999       4.3G    0.02318    0.01011   0.007391         32        640: 100% 32/32 [00:10<00:00,  3.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.58it/s]
                   all        129        138      0.542       0.45       0.46      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   118/3999       4.3G    0.02157    0.01006   0.007071         21        640: 100% 32/32 [00:06<00:00,  4.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.60it/s]
                   all        129        138      0.555      0.519      0.501      0.235

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   119/3999       4.3G    0.02106    0.01038   0.006557         27        640: 100% 32/32 [00:08<00:00,  3.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.91it/s]
                   all        129        138      0.523      0.445      0.428       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   120/3999       4.3G    0.02126   0.009926   0.006792         30        640: 100% 32/32 [00:06<00:00,  4.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.05it/s]
                   all        129        138      0.507      0.472      0.465      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   121/3999       4.3G    0.02208    0.01005   0.005984         34        640: 100% 32/32 [00:07<00:00,  4.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.83it/s]
                   all        129        138      0.557      0.516      0.452      0.209

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   122/3999       4.3G     0.0205    0.00969    0.00657         32        640: 100% 32/32 [00:07<00:00,  4.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.75it/s]
                   all        129        138      0.551      0.504      0.435      0.211

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   123/3999       4.3G    0.02203    0.01023   0.006212         34        640: 100% 32/32 [00:06<00:00,  4.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.68it/s]
                   all        129        138      0.546      0.495      0.426      0.197

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   124/3999       4.3G    0.02045   0.009944   0.006453         27        640: 100% 32/32 [00:08<00:00,  3.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.02it/s]
                   all        129        138      0.499      0.516       0.43      0.183

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   125/3999       4.3G    0.02231    0.01018   0.006811         28        640: 100% 32/32 [00:06<00:00,  4.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.68it/s]
                   all        129        138      0.486      0.516      0.462      0.207

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   126/3999       4.3G    0.02136   0.009937   0.005936         26        640: 100% 32/32 [00:08<00:00,  3.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.57it/s]
                   all        129        138      0.538      0.505      0.453       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   127/3999       4.3G    0.02055   0.009856   0.007385         35        640: 100% 32/32 [00:06<00:00,  4.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.61it/s]
                   all        129        138      0.538      0.489      0.452      0.205

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   128/3999       4.3G    0.02111     0.0093   0.005777         24        640: 100% 32/32 [00:07<00:00,  4.06it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.70it/s]
                   all        129        138       0.54      0.498      0.491      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   129/3999       4.3G    0.02199   0.009084   0.006903         40        640: 100% 32/32 [00:07<00:00,  4.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.72it/s]
                   all        129        138      0.602      0.525      0.519      0.244

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   130/3999       4.3G    0.02108   0.009678   0.007326         24        640: 100% 32/32 [00:06<00:00,  4.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.629      0.501      0.531      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   131/3999       4.3G    0.02202   0.009668   0.006655         36        640: 100% 32/32 [00:07<00:00,  4.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.88it/s]
                   all        129        138      0.654      0.539      0.545       0.27

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   132/3999       4.3G    0.02168    0.00964   0.007053         29        640: 100% 32/32 [00:06<00:00,  4.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.66it/s]
                   all        129        138      0.494      0.523       0.49      0.221

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   133/3999       4.3G    0.02077   0.009448   0.006062         22        640: 100% 32/32 [00:08<00:00,  3.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.54it/s]
                   all        129        138      0.557      0.554      0.556      0.251

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   134/3999       4.3G    0.02095    0.00951   0.005716         23        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.40it/s]
                   all        129        138      0.572      0.535      0.515      0.236

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   135/3999       4.3G    0.02119   0.009392   0.005581         32        640: 100% 32/32 [00:08<00:00,  3.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.75it/s]
                   all        129        138      0.518      0.569       0.52      0.243

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   136/3999       4.3G    0.02082   0.009528   0.006299         26        640: 100% 32/32 [00:06<00:00,  4.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.33it/s]
                   all        129        138      0.594      0.451      0.496      0.223

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   137/3999       4.3G    0.02195   0.009354   0.005562         25        640: 100% 32/32 [00:07<00:00,  4.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.609       0.56      0.532      0.234

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   138/3999       4.3G    0.02036   0.009499   0.004873         21        640: 100% 32/32 [00:07<00:00,  4.47it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.95it/s]
                   all        129        138      0.499      0.529      0.539      0.253

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   139/3999       4.3G    0.02112   0.009399   0.006039         35        640: 100% 32/32 [00:07<00:00,  4.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.86it/s]
                   all        129        138       0.56      0.568      0.519      0.246

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   140/3999       4.3G    0.01923   0.008883   0.007033         30        640: 100% 32/32 [00:09<00:00,  3.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:02<00:00,  2.37it/s]
                   all        129        138      0.529      0.492      0.466      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   141/3999       4.3G     0.0198   0.009361   0.005675         28        640: 100% 32/32 [00:06<00:00,  4.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.66it/s]
                   all        129        138      0.491      0.558      0.469      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   142/3999       4.3G     0.0203   0.009297   0.005775         26        640: 100% 32/32 [00:08<00:00,  3.94it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.23it/s]
                   all        129        138      0.497      0.479      0.485      0.235

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   143/3999       4.3G    0.02013   0.009262   0.006513         28        640: 100% 32/32 [00:06<00:00,  4.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.501      0.471       0.47      0.218

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   144/3999       4.3G    0.02097   0.009267   0.005235         28        640: 100% 32/32 [00:08<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.47it/s]
                   all        129        138      0.581      0.458       0.46      0.214

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   145/3999       4.3G    0.02066   0.009513   0.007048         28        640: 100% 32/32 [00:06<00:00,  4.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.588      0.535      0.505      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   146/3999       4.3G    0.01942   0.009289   0.006269         32        640: 100% 32/32 [00:08<00:00,  3.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.84it/s]
                   all        129        138      0.559      0.567      0.521      0.246

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   147/3999       4.3G    0.01902   0.008427   0.005065         32        640: 100% 32/32 [00:06<00:00,  4.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.18it/s]
                   all        129        138      0.487      0.495      0.471      0.233

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   148/3999       4.3G    0.02035   0.008869   0.005094         25        640: 100% 32/32 [00:07<00:00,  4.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.83it/s]
                   all        129        138      0.611      0.414      0.471      0.208

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   149/3999       4.3G    0.02008   0.008838   0.004814         44        640: 100% 32/32 [00:07<00:00,  4.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.84it/s]
                   all        129        138      0.515      0.488      0.452      0.208

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   150/3999       4.3G    0.02051   0.009102   0.005316         29        640: 100% 32/32 [00:06<00:00,  4.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.83it/s]
                   all        129        138      0.616      0.562      0.534      0.244

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   151/3999       4.3G    0.02014   0.009469   0.005778         29        640: 100% 32/32 [00:08<00:00,  3.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.13it/s]
                   all        129        138      0.614      0.592      0.551      0.259

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   152/3999       4.3G    0.02102   0.009616   0.005589         32        640: 100% 32/32 [00:06<00:00,  4.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.568      0.599      0.535      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   153/3999       4.3G    0.02078   0.009573   0.006799         33        640: 100% 32/32 [00:08<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.56it/s]
                   all        129        138      0.531      0.464      0.474      0.207

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   154/3999       4.3G    0.01961   0.009176   0.006036         26        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.87it/s]
                   all        129        138       0.58      0.412      0.455      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   155/3999       4.3G    0.02072   0.009777   0.004662         35        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.82it/s]
                   all        129        138      0.631      0.471      0.515      0.236

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   156/3999       4.3G    0.01971   0.009288   0.007473         23        640: 100% 32/32 [00:06<00:00,  4.90it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.43it/s]
                   all        129        138      0.547      0.559      0.535      0.247

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   157/3999       4.3G    0.01926   0.009602   0.005432         32        640: 100% 32/32 [00:07<00:00,  4.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.70it/s]
                   all        129        138      0.546      0.516      0.475      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   158/3999       4.3G    0.01937    0.01006   0.006984         30        640: 100% 32/32 [00:07<00:00,  4.46it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.89it/s]
                   all        129        138      0.557      0.467      0.451      0.212

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   159/3999       4.3G    0.02144    0.00956   0.006914         27        640: 100% 32/32 [00:06<00:00,  4.57it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.71it/s]
                   all        129        138      0.479      0.481      0.472      0.209

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   160/3999       4.3G    0.02097   0.009141   0.006041         24        640: 100% 32/32 [00:07<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.82it/s]
                   all        129        138      0.433      0.434       0.38      0.157

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   161/3999       4.3G    0.02072   0.009745   0.008682         30        640: 100% 32/32 [00:06<00:00,  4.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.87it/s]
                   all        129        138      0.489      0.433      0.408      0.169

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   162/3999       4.3G     0.0192   0.009113   0.005484         24        640: 100% 32/32 [00:08<00:00,  3.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.64it/s]
                   all        129        138       0.52      0.456      0.439      0.204

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   163/3999       4.3G    0.02066   0.009678   0.006209         23        640: 100% 32/32 [00:07<00:00,  4.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.52it/s]
                   all        129        138      0.568      0.548      0.527       0.26

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   164/3999       4.3G     0.0193   0.009032   0.005907         30        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.558      0.494      0.512      0.245

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   165/3999       4.3G    0.02044    0.00914   0.004711         28        640: 100% 32/32 [00:06<00:00,  4.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.08it/s]
                   all        129        138      0.525      0.505      0.464      0.223

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   166/3999       4.3G     0.0196   0.009397   0.004781         29        640: 100% 32/32 [00:07<00:00,  4.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.59it/s]
                   all        129        138      0.521      0.532        0.5       0.24

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   167/3999       4.3G    0.02095   0.008761   0.005491         27        640: 100% 32/32 [00:07<00:00,  4.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.93it/s]
                   all        129        138      0.466      0.484      0.467      0.215

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   168/3999       4.3G    0.02036   0.009059   0.004909         32        640: 100% 32/32 [00:06<00:00,  4.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.477      0.469      0.488      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   169/3999       4.3G    0.01974    0.00898   0.004949         29        640: 100% 32/32 [00:08<00:00,  3.90it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.52it/s]
                   all        129        138      0.509      0.488      0.504      0.246

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   170/3999       4.3G    0.01975    0.00901   0.005266         26        640: 100% 32/32 [00:06<00:00,  4.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.67it/s]
                   all        129        138      0.552      0.505       0.48      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   171/3999       4.3G    0.01921   0.008976   0.004946         17        640: 100% 32/32 [00:08<00:00,  3.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.42it/s]
                   all        129        138      0.523      0.535      0.508      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   172/3999       4.3G    0.02026   0.009055    0.00522         32        640: 100% 32/32 [00:06<00:00,  4.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.39it/s]
                   all        129        138      0.558      0.535      0.541      0.224

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   173/3999       4.3G    0.01919   0.009409   0.006358         33        640: 100% 32/32 [00:08<00:00,  3.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.70it/s]
                   all        129        138      0.465       0.52      0.484      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   174/3999       4.3G    0.01966   0.009295   0.006051         30        640: 100% 32/32 [00:06<00:00,  4.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.37it/s]
                   all        129        138      0.468      0.486      0.441      0.198

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   175/3999       4.3G    0.01971   0.008909   0.006777         32        640: 100% 32/32 [00:07<00:00,  4.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.87it/s]
                   all        129        138      0.515      0.478      0.434      0.183

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   176/3999       4.3G    0.01877   0.009177   0.005947         30        640: 100% 32/32 [00:07<00:00,  4.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.78it/s]
                   all        129        138      0.533      0.504      0.487      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   177/3999       4.3G    0.01976   0.008669   0.005423         36        640: 100% 32/32 [00:06<00:00,  4.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.83it/s]
                   all        129        138       0.58      0.591       0.51      0.246

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   178/3999       4.3G     0.0184   0.008673   0.004507         32        640: 100% 32/32 [00:08<00:00,  3.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.69it/s]
                   all        129        138      0.559       0.52      0.515      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   179/3999       4.3G    0.01853   0.008883   0.004388         29        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.541       0.52      0.508       0.25

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   180/3999       4.3G    0.01968   0.008326   0.005325         20        640: 100% 32/32 [00:08<00:00,  3.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.67it/s]
                   all        129        138      0.514      0.479      0.464      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   181/3999       4.3G    0.01984   0.009054   0.004646         29        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.62it/s]
                   all        129        138      0.573      0.442       0.47      0.229

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   182/3999       4.3G    0.01916   0.008876    0.00386         33        640: 100% 32/32 [00:08<00:00,  3.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.81it/s]
                   all        129        138      0.444      0.513       0.43       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   183/3999       4.3G    0.01944   0.008768   0.005339         25        640: 100% 32/32 [00:06<00:00,  4.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.32it/s]
                   all        129        138      0.502      0.489       0.45       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   184/3999       4.3G    0.01955   0.009279    0.00748         29        640: 100% 32/32 [00:07<00:00,  4.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.68it/s]
                   all        129        138       0.58      0.487      0.476      0.243

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   185/3999       4.3G    0.01903   0.008998   0.006055         27        640: 100% 32/32 [00:07<00:00,  4.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.80it/s]
                   all        129        138      0.621      0.485      0.499      0.256

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   186/3999       4.3G    0.01868   0.008733   0.004864         30        640: 100% 32/32 [00:07<00:00,  4.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.94it/s]
                   all        129        138      0.615      0.469      0.503       0.26

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   187/3999       4.3G    0.01947   0.008948   0.005707         29        640: 100% 32/32 [00:09<00:00,  3.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.46it/s]
                   all        129        138      0.588      0.486      0.516      0.257

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   188/3999       4.3G    0.01893    0.00801   0.005267         30        640: 100% 32/32 [00:06<00:00,  4.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.65it/s]
                   all        129        138      0.565      0.606      0.559      0.277

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   189/3999       4.3G    0.01806   0.008792   0.004282         28        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.526      0.587      0.535       0.26

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   190/3999       4.3G    0.01961   0.008732   0.004315         28        640: 100% 32/32 [00:06<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.18it/s]
                   all        129        138      0.546      0.566      0.541      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   191/3999       4.3G     0.0178    0.00897   0.004928         30        640: 100% 32/32 [00:07<00:00,  4.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.537      0.627      0.537      0.233

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   192/3999       4.3G    0.01997   0.008134   0.004136         22        640: 100% 32/32 [00:07<00:00,  4.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.02it/s]
                   all        129        138      0.622      0.509      0.551      0.243

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   193/3999       4.3G    0.01849   0.008498   0.005477         35        640: 100% 32/32 [00:06<00:00,  4.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.557      0.539      0.508      0.232

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   194/3999       4.3G    0.01859   0.008913   0.004743         33        640: 100% 32/32 [00:08<00:00,  3.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.47it/s]
                   all        129        138      0.536      0.521      0.505      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   195/3999       4.3G     0.0194   0.008936   0.005429         32        640: 100% 32/32 [00:06<00:00,  4.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.66it/s]
                   all        129        138      0.459      0.561      0.457      0.221

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   196/3999       4.3G    0.01896   0.007905    0.00553         32        640: 100% 32/32 [00:08<00:00,  3.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.61it/s]
                   all        129        138      0.445      0.487      0.414      0.183

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   197/3999       4.3G    0.01849   0.008671    0.00387         31        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.86it/s]
                   all        129        138      0.493      0.488      0.445      0.196

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   198/3999       4.3G    0.01896   0.008921   0.005152         31        640: 100% 32/32 [00:08<00:00,  3.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.442      0.467      0.431      0.202

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   199/3999       4.3G    0.01774   0.008508   0.004361         25        640: 100% 32/32 [00:06<00:00,  4.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.17it/s]
                   all        129        138      0.553      0.496      0.465      0.227

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   200/3999       4.3G    0.01905   0.008297   0.004602         24        640: 100% 32/32 [00:07<00:00,  4.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.61it/s]
                   all        129        138       0.61       0.47      0.499      0.236

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   201/3999       4.3G    0.01814   0.008307   0.004885         28        640: 100% 32/32 [00:07<00:00,  4.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.82it/s]
                   all        129        138      0.552      0.547      0.512      0.236

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   202/3999       4.3G    0.02016   0.008683   0.005835         25        640: 100% 32/32 [00:06<00:00,  4.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.79it/s]
                   all        129        138      0.579      0.526      0.526       0.24

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   203/3999       4.3G    0.01888   0.008257   0.005025         22        640: 100% 32/32 [00:08<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.71it/s]
                   all        129        138      0.547      0.475      0.482      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   204/3999       4.3G    0.01815   0.009054   0.004418         29        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.82it/s]
                   all        129        138      0.514      0.548       0.47       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   205/3999       4.3G    0.01864   0.008821   0.004451         35        640: 100% 32/32 [00:08<00:00,  3.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.67it/s]
                   all        129        138      0.555      0.545      0.517      0.227

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   206/3999       4.3G    0.01756   0.008509   0.005225         30        640: 100% 32/32 [00:06<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.81it/s]
                   all        129        138      0.553      0.488      0.481       0.23

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   207/3999       4.3G     0.0201   0.008956    0.00523         28        640: 100% 32/32 [00:08<00:00,  3.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.59it/s]
                   all        129        138      0.537      0.452      0.467      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   208/3999       4.3G    0.01952   0.008549   0.005753         37        640: 100% 32/32 [00:06<00:00,  4.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.11it/s]
                   all        129        138      0.486       0.51      0.472      0.221

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   209/3999       4.3G    0.01886   0.008927   0.005004         31        640: 100% 32/32 [00:07<00:00,  4.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.85it/s]
                   all        129        138      0.645      0.498      0.534      0.241

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   210/3999       4.3G    0.01762   0.007992   0.004273         30        640: 100% 32/32 [00:08<00:00,  3.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.48it/s]
                   all        129        138      0.521        0.5      0.463      0.237

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   211/3999       4.3G     0.0184    0.00847   0.005156         25        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.84it/s]
                   all        129        138      0.555      0.554      0.505      0.233

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   212/3999       4.3G    0.01952   0.008193   0.004572         24        640: 100% 32/32 [00:08<00:00,  3.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.59it/s]
                   all        129        138      0.589      0.491      0.503      0.237

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   213/3999       4.3G    0.01879   0.008502   0.004192         27        640: 100% 32/32 [00:06<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.51it/s]
                   all        129        138      0.525      0.465      0.451      0.204

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   214/3999       4.3G    0.01889   0.008442   0.004546         33        640: 100% 32/32 [00:07<00:00,  4.00it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.82it/s]
                   all        129        138      0.557      0.531      0.506       0.22

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   215/3999       4.3G      0.019   0.008594   0.005488         23        640: 100% 32/32 [00:07<00:00,  4.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.76it/s]
                   all        129        138      0.666      0.495      0.512      0.228

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   216/3999       4.3G    0.01867   0.009105   0.005324         29        640: 100% 32/32 [00:07<00:00,  4.45it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.78it/s]
                   all        129        138      0.524       0.52      0.483      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   217/3999       4.3G    0.01905   0.008677    0.00537         32        640: 100% 32/32 [00:07<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.66it/s]
                   all        129        138      0.597      0.417      0.442      0.206

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   218/3999       4.3G    0.01858    0.00854   0.004603         29        640: 100% 32/32 [00:06<00:00,  4.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.84it/s]
                   all        129        138      0.549      0.432      0.402      0.191

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   219/3999       4.3G    0.01772   0.008795   0.004845         30        640: 100% 32/32 [00:08<00:00,  3.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.33it/s]
                   all        129        138      0.589      0.483      0.451       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   220/3999       4.3G    0.01859   0.008189   0.005286         29        640: 100% 32/32 [00:06<00:00,  4.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.505      0.502      0.469      0.233

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   221/3999       4.3G    0.01934   0.008862   0.005536         24        640: 100% 32/32 [00:08<00:00,  3.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.91it/s]
                   all        129        138      0.523      0.444       0.42      0.201

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   222/3999       4.3G    0.01866   0.008518   0.003931         37        640: 100% 32/32 [00:06<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.52it/s]
                   all        129        138      0.486      0.497      0.464      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   223/3999       4.3G    0.01923   0.008251   0.004501         19        640: 100% 32/32 [00:07<00:00,  4.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.77it/s]
                   all        129        138      0.599      0.515      0.498      0.242

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   224/3999       4.3G    0.01826   0.008622   0.005793         18        640: 100% 32/32 [00:07<00:00,  4.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.74it/s]
                   all        129        138       0.54        0.5      0.501      0.253

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   225/3999       4.3G    0.01802   0.008725   0.005037         32        640: 100% 32/32 [00:07<00:00,  4.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.75it/s]
                   all        129        138      0.573      0.474      0.526      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   226/3999       4.3G    0.01686    0.00806   0.004243         28        640: 100% 32/32 [00:07<00:00,  4.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.18it/s]
                   all        129        138      0.575      0.537      0.528      0.249

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   227/3999       4.3G    0.01803   0.008754   0.005058         27        640: 100% 32/32 [00:06<00:00,  4.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.67it/s]
                   all        129        138      0.563      0.494      0.504      0.227

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   228/3999       4.3G    0.01687   0.008564   0.003759         33        640: 100% 32/32 [00:08<00:00,  3.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.25it/s]
                   all        129        138      0.576       0.54      0.538       0.23

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   229/3999       4.3G    0.01834   0.008484   0.005969         31        640: 100% 32/32 [00:06<00:00,  4.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.50it/s]
                   all        129        138      0.568      0.527      0.534      0.239

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   230/3999       4.3G    0.01847   0.008188   0.002991         35        640: 100% 32/32 [00:08<00:00,  3.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.65it/s]
                   all        129        138      0.515      0.544      0.512       0.22

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   231/3999       4.3G    0.01791   0.008159   0.004195         33        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.45it/s]
                   all        129        138      0.576      0.528      0.533       0.25

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   232/3999       4.3G    0.01807   0.008536   0.005013         31        640: 100% 32/32 [00:08<00:00,  3.90it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.72it/s]
                   all        129        138      0.564      0.522      0.515      0.245

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   233/3999       4.3G    0.01808   0.008197   0.003484         34        640: 100% 32/32 [00:08<00:00,  3.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.76it/s]
                   all        129        138      0.542      0.547      0.488      0.233

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   234/3999       4.3G    0.01682   0.008081    0.00486         29        640: 100% 32/32 [00:06<00:00,  4.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.86it/s]
                   all        129        138       0.61      0.474      0.534      0.263

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   235/3999       4.3G    0.01748   0.008465   0.004492         30        640: 100% 32/32 [00:08<00:00,  3.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.651      0.492      0.532      0.271

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   236/3999       4.3G    0.01739   0.008526   0.004944         21        640: 100% 32/32 [00:06<00:00,  4.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.78it/s]
                   all        129        138      0.616      0.546       0.54      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   237/3999       4.3G    0.01747   0.007781   0.005714         26        640: 100% 32/32 [00:08<00:00,  3.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.26it/s]
                   all        129        138      0.544      0.497      0.484      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   238/3999       4.3G    0.01746   0.008391   0.004051         37        640: 100% 32/32 [00:06<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.32it/s]
                   all        129        138      0.521      0.558      0.476        0.2

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   239/3999       4.3G    0.02058   0.008156   0.005296         34        640: 100% 32/32 [00:07<00:00,  4.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.85it/s]
                   all        129        138      0.492      0.516      0.419      0.182

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   240/3999       4.3G    0.01845   0.008327   0.004447         29        640: 100% 32/32 [00:07<00:00,  4.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.73it/s]
                   all        129        138      0.476      0.503      0.439      0.193

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   241/3999       4.3G    0.01967   0.007768   0.003686         26        640: 100% 32/32 [00:06<00:00,  4.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.88it/s]
                   all        129        138       0.57      0.433      0.459       0.21

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   242/3999       4.3G    0.01747   0.008166   0.003423         25        640: 100% 32/32 [00:07<00:00,  4.02it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.02it/s]
                   all        129        138       0.55      0.529      0.512      0.237

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   243/3999       4.3G    0.01908     0.0084   0.005132         34        640: 100% 32/32 [00:06<00:00,  4.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.85it/s]
                   all        129        138      0.508      0.525      0.511      0.242

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   244/3999       4.3G    0.01786   0.007844   0.003718         31        640: 100% 32/32 [00:08<00:00,  3.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.63it/s]
                   all        129        138      0.455      0.567      0.459      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   245/3999       4.3G    0.01695   0.008352   0.003734         29        640: 100% 32/32 [00:06<00:00,  4.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.81it/s]
                   all        129        138      0.491      0.542      0.491      0.243

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   246/3999       4.3G    0.01832   0.008055   0.003246         23        640: 100% 32/32 [00:08<00:00,  3.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.75it/s]
                   all        129        138      0.605      0.456      0.505      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   247/3999       4.3G    0.01753   0.008506   0.003956         34        640: 100% 32/32 [00:06<00:00,  4.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.39it/s]
                   all        129        138      0.591      0.433      0.484      0.236

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   248/3999       4.3G     0.0182   0.008079   0.003752         25        640: 100% 32/32 [00:07<00:00,  4.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.522      0.459      0.469      0.235

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   249/3999       4.3G    0.01776   0.008043   0.003217         21        640: 100% 32/32 [00:07<00:00,  4.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.75it/s]
                   all        129        138      0.511      0.474      0.414      0.192

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   250/3999       4.3G    0.01815   0.008315   0.003712         26        640: 100% 32/32 [00:06<00:00,  4.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.83it/s]
                   all        129        138      0.615       0.44      0.494      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   251/3999       4.3G    0.01693   0.007887   0.004782         27        640: 100% 32/32 [00:08<00:00,  3.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.85it/s]
                   all        129        138      0.484      0.454       0.45      0.206

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   252/3999       4.3G    0.01706   0.008171   0.003473         23        640: 100% 32/32 [00:06<00:00,  4.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.84it/s]
                   all        129        138      0.554      0.531      0.504      0.239

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   253/3999       4.3G    0.01719    0.00787    0.00747         34        640: 100% 32/32 [00:08<00:00,  3.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.76it/s]
                   all        129        138      0.495      0.508      0.488       0.22

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   254/3999       4.3G    0.01714   0.008931   0.003897         22        640: 100% 32/32 [00:06<00:00,  4.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.31it/s]
                   all        129        138       0.57      0.556      0.526      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   255/3999       4.3G    0.01957   0.008508   0.006562         28        640: 100% 32/32 [00:10<00:00,  3.15it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.66it/s]
                   all        129        138      0.533      0.535      0.486      0.218

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   256/3999       4.3G    0.01677   0.008313   0.004432         31        640: 100% 32/32 [00:07<00:00,  4.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.87it/s]
                   all        129        138       0.49      0.464      0.422      0.197

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   257/3999       4.3G    0.01825   0.008637   0.004677         31        640: 100% 32/32 [00:06<00:00,  4.58it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.85it/s]
                   all        129        138      0.476      0.479      0.477      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   258/3999       4.3G    0.01819   0.008755   0.004591         27        640: 100% 32/32 [00:08<00:00,  3.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.37it/s]
                   all        129        138      0.578      0.423      0.449      0.208

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   259/3999       4.3G     0.0164   0.007792   0.004454         20        640: 100% 32/32 [00:06<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.81it/s]
                   all        129        138      0.554      0.516      0.529      0.243

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   260/3999       4.3G    0.01773   0.008386   0.003886         31        640: 100% 32/32 [00:08<00:00,  3.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.44it/s]
                   all        129        138      0.528      0.458      0.481      0.212

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   261/3999       4.3G    0.01815   0.008145     0.0049         32        640: 100% 32/32 [00:06<00:00,  4.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.45it/s]
                   all        129        138      0.516      0.479      0.489      0.227

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   262/3999       4.3G    0.01745   0.007914   0.004165         28        640: 100% 32/32 [00:07<00:00,  4.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.79it/s]
                   all        129        138      0.611      0.487      0.519      0.223

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   263/3999       4.3G      0.017   0.007963   0.005517         33        640: 100% 32/32 [00:06<00:00,  4.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.56it/s]
                   all        129        138      0.559       0.53        0.5      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   264/3999       4.3G    0.01628   0.007911   0.004112         25        640: 100% 32/32 [00:07<00:00,  4.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.61it/s]
                   all        129        138      0.518      0.527      0.484      0.225

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   265/3999       4.3G    0.01714   0.008017   0.003485         30        640: 100% 32/32 [00:07<00:00,  4.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.95it/s]
                   all        129        138      0.509      0.488      0.482      0.214

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   266/3999       4.3G    0.01742   0.008035   0.004089         35        640: 100% 32/32 [00:06<00:00,  4.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.77it/s]
                   all        129        138       0.53      0.463      0.475      0.213

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   267/3999       4.3G    0.01701   0.007768   0.003745         34        640: 100% 32/32 [00:08<00:00,  3.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.59it/s]
                   all        129        138      0.505      0.537      0.446      0.212

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   268/3999       4.3G    0.01708   0.007643   0.003426         33        640: 100% 32/32 [00:06<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.78it/s]
                   all        129        138      0.502      0.535      0.454      0.217

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   269/3999       4.3G    0.01859   0.008046    0.00283         21        640: 100% 32/32 [00:08<00:00,  3.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.68it/s]
                   all        129        138      0.575      0.539      0.502      0.252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   270/3999       4.3G    0.01641   0.008063   0.003666         26        640: 100% 32/32 [00:06<00:00,  4.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.34it/s]
                   all        129        138      0.527      0.572      0.544      0.259

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   271/3999       4.3G    0.01621   0.007517   0.005019         27        640: 100% 32/32 [00:07<00:00,  4.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138      0.537      0.486      0.521      0.245

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   272/3999       4.3G    0.01677   0.007707   0.003329         29        640: 100% 32/32 [00:07<00:00,  4.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.65it/s]
                   all        129        138      0.657      0.452      0.501      0.216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   273/3999       4.3G    0.01581   0.007765   0.005203         26        640: 100% 32/32 [00:07<00:00,  4.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.81it/s]
                   all        129        138      0.682      0.487       0.51      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   274/3999       4.3G    0.01797    0.00812   0.003327         23        640: 100% 32/32 [00:08<00:00,  3.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.11it/s]
                   all        129        138      0.516      0.532      0.496      0.233

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   275/3999       4.3G    0.01632   0.007819   0.004366         30        640: 100% 32/32 [00:06<00:00,  4.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.78it/s]
                   all        129        138      0.667      0.518      0.542       0.25

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   276/3999       4.3G    0.01685   0.007869   0.003417         23        640: 100% 32/32 [00:08<00:00,  3.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.59it/s]
                   all        129        138      0.544      0.485      0.458      0.199

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   277/3999       4.3G    0.01844   0.008217   0.003815         29        640: 100% 32/32 [00:06<00:00,  4.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.73it/s]
                   all        129        138      0.581      0.436      0.446      0.201

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   278/3999       4.3G    0.01768   0.008034   0.004069         31        640: 100% 32/32 [00:10<00:00,  3.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.54it/s]
                   all        129        138      0.583      0.495        0.5      0.222

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   279/3999       4.3G    0.01636   0.007597   0.004342         31        640: 100% 32/32 [00:06<00:00,  4.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.77it/s]
                   all        129        138      0.577      0.515      0.516      0.241

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   280/3999       4.3G    0.01674   0.007695   0.004678         21        640: 100% 32/32 [00:08<00:00,  3.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.72it/s]
                   all        129        138       0.54      0.531      0.507       0.23

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   281/3999       4.3G    0.01642   0.007801   0.003907         33        640: 100% 32/32 [00:06<00:00,  4.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.15it/s]
                   all        129        138      0.576      0.495      0.498      0.245

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   282/3999       4.3G    0.01634   0.007426   0.003423         37        640: 100% 32/32 [00:07<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.77it/s]
                   all        129        138      0.601      0.531      0.536      0.256

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   283/3999       4.3G    0.01643   0.007622     0.0042         30        640: 100% 32/32 [00:07<00:00,  4.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.61it/s]
                   all        129        138      0.586      0.527      0.516      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   284/3999       4.3G    0.01634   0.007828   0.003834         27        640: 100% 32/32 [00:06<00:00,  4.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.86it/s]
                   all        129        138      0.644      0.486      0.502      0.235

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   285/3999       4.3G     0.0169   0.008212   0.006798         29        640: 100% 32/32 [00:08<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.88it/s]
                   all        129        138      0.529      0.493      0.508      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   286/3999       4.3G    0.01711   0.007511   0.003781         24        640: 100% 32/32 [00:06<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.69it/s]
                   all        129        138      0.582      0.532      0.473      0.226

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   287/3999       4.3G    0.01669   0.007839   0.004715         33        640: 100% 32/32 [00:08<00:00,  3.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.62it/s]
                   all        129        138      0.576      0.547      0.482      0.235

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   288/3999       4.3G    0.01713   0.007265   0.004018         33        640: 100% 32/32 [00:06<00:00,  4.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  4.26it/s]
                   all        129        138      0.587      0.559      0.544       0.26
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 188, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

289 epochs completed in 0.741 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.5MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.5MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model summary: 157 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:04<00:00,  1.06it/s]
                   all        129        138      0.569       0.61      0.559      0.277
           laying down        129         48      0.447      0.555      0.427       0.18
               sitting        129         44       0.55      0.472      0.426      0.202
             squatting        129         46      0.709      0.804      0.824      0.449
Results saved to runs/train/exp
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:      metrics/mAP_0.5 ▁▄▅▇▅▇▄▇▆▅▇▅▆▆▇▇▅▅█▇▆▇█▆▅▅█▆▆▇▅▇█▇▆▆▇▇▇▆
wandb: metrics/mAP_0.5:0.95 ▁▃▄▅▅▆▄▆▅▄▆▄▆▆▆▆▅▄█▇▆▆▇▅▄▅▆▆▅▆▅▆█▇▆▅▆▇▇▆
wandb:    metrics/precision ▂▃▄▅▄▅▁▆▄▄▆▅▅▇▆▅▅▆█▆▄▇▆▄▅▅▅▆▆▆▇▆█▅▇▅▆▅▆▆
wandb:       metrics/recall ▁▅▅▇▅▆▅▆▃▅▆▅▆▄▇▆▄▅▆▇▅▅▆▅▅▅█▅▅▆▅▅▅▆▄▆▆▅▆▆
wandb:       train/box_loss █▅▄▄▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁
wandb:       train/cls_loss █▇▆▅▄▄▃▃▃▃▃▃▂▂▂▂▂▂▂▂▁▁▂▂▂▂▁▁▁▁▁▁▁▁▁▂▂▁▁▁
wandb:       train/obj_loss █▇▆▆▅▄▄▄▄▃▃▃▃▃▂▂▃▃▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▁▁▁▁
wandb:         val/box_loss █▆▃▃▃▂▃▂▂▃▂▃▃▂▁▂▂▃▁▂▃▂▂▂▂▄▂▂▂▂▂▂▂▂▁▃▁▂▂▃
wandb:         val/cls_loss ▅▄▅▄▄▂▅▄▅▃▃▃▃▃▂▃▆▄▂▃▃▄▂▂▃▃▂▃▄▁▂▂▂█▇▃▃▄█▃
wandb:         val/obj_loss ▁▁▂▃▄▂▆▁▄▆▄▃▃▃▃▂█▅▄▆▆▅▂▃▄▄▄█▅█▅▅▃▄▄▃▅▅▄▄
wandb:                x/lr0 ████▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                x/lr1 ████▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                x/lr2 ████▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb: 
wandb: Run summary:
wandb:           best/epoch 188
wandb:         best/mAP_0.5 0.55877
wandb:    best/mAP_0.5:0.95 0.27717
wandb:       best/precision 0.56518
wandb:          best/recall 0.60613
wandb:      metrics/mAP_0.5 0.55874
wandb: metrics/mAP_0.5:0.95 0.27718
wandb:    metrics/precision 0.5686
wandb:       metrics/recall 0.6104
wandb:       train/box_loss 0.01713
wandb:       train/cls_loss 0.00402
wandb:       train/obj_loss 0.00726
wandb:         val/box_loss 0.0233
wandb:         val/cls_loss 0.00939
wandb:         val/obj_loss 0.00815
wandb:                x/lr0 0.00929
wandb:                x/lr1 0.00929
wandb:                x/lr2 0.00929
wandb: 
wandb: 🚀 View run rich-haze-1 at: https://wandb.ai/emgs/YOLOv5/runs/19jirz9b
wandb: Synced 5 W&B file(s), 17 media file(s), 3 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230430_183631-19jirz9b/logs
wandb: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
  ```
</details>



