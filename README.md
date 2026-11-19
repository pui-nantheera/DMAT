# DMAT: An End-to-End Framework for Joint Atmospheric Turbulence Mitigation and Object Detection [WACV2026]

[[Paper](https://arxiv.org/pdf/2507.04323)] [[Real dataset](https://zenodo.org/records/13737763)] [[Synthetic datasets](https://zenodo.org/records/17673509)] [[Pretrained models](https://zenodo.org/records/17673509)]

## Description
This code contains a linked turbulence mitigation model (based on VideoMamba and TMT) together with object detection code (based on DETR and LWDETR). 

## Installation
This project is developed within a Ubuntu 22.04 OS docker image (using Python 3.10.12 and CUDA 12.4). You will therefore need to have docker installed. 

### Docker installation
There are many different ways to install docker according to your set up (WSL, Windows, Linux etc.).  However general guides are here as you should aim to enable CUDA within containers.

* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.8.1/install-guide.html
* https://medium.com/@albertqueralto/enabling-cuda-capabilities-in-docker-containers-51a3566ad014


## Docker
Docker image is defined in the dockerfile

### Build docker image

Firstly you need to build the image
```bash
sudo docker build -t turb_mit .
```

### Run docker image
Secondly run the docker image and drop into a bash script and change into the code directory

```bash
sudo docker run --shm-size=8g -v ./code:/app/code -it --gpus all turb_detect bash 
```

When you have done this, you will also have to fo the following instalation
```bash
cd Deformable-DETR/models/ops
pip install -e .
cd ../../..
```

## Training

```bash
python train.py --batch_size 1 \
--run_name "detr_medium_static" \
--patch_size 256 \
--train_path "./dataset_example/DYNAMIC_1/train2017carperson" \
--val_path "./dataset_example/DYNAMIC_1/val2017carperson" \
--log_path "./log" \
--train_ann_file "./dataset_example/coco_processed/annotations/instances_train2017_crop_carperson.json" \
--train_im_dir ./dataset_example/coco_processed/images/train2017carperson \
--val_ann_file "./dataset_example/coco_processed/annotations/instances_val2017_crop_carperson.json" \
--val_im_dir "./dataset_example/coco_processed/images/val2017carperson" \
--load_tmt "./tmt_checkpoints/turb_model_best.pth" \
--start_over

python train.py --batch_size 1 \
--run_name "Lwdetr_medium_static" \
--patch_size 256 \
--train_path "./dataset_example/DYNAMIC_1/train2017carperson" \
--val_path "./dataset_example/DYNAMIC_1/val2017carperson" \
--log_path "./log" \
--train_ann_file "./dataset_example/coco_processed/annotations/instances_train2017_crop_carperson.json" \
--train_im_dir ./dataset_example/coco_processed/images/train2017carperson \
--val_ann_file "./dataset_example/coco_processed/annotations/instances_val2017_crop_carperson.json" \
--val_im_dir "./dataset_example/coco_processed/images/val2017carperson" \
--load_tmt "./tmt_checkpoints/turb_model_best.pth" \
--lwdetr \
--lwdetr_model "detr_medium" \
--start_over

python train.py --batch_size 1 \
--run_name "Detr_medium_dyn" \
--patch_size 256 \
--log_path "./log" \
--load_tmt "./tmt_checkpoints/turb_model_best_dyn.pth" \
--train_path "/home/csprh/scratch/COMBO_COCO_DATA/STATIC_GT_P2S_MP4S/traindyn" \
--val_path "/home/csprh/scratch/COMBO_COCO_DATA/STATIC_GT_P2S_MP4S/valdyn" \
--train_ann_file "/home/csprh/scratch/GOT10k/got10k_filtered/annotations/instances_train2017.json" \
--train_im_dir "/home/csprh/scratch/GOT10k/got10k_filtered/images/train2017" \
--val_ann_file "/home/csprh/scratch/GOT10k/got10k_filtered/annotations/instances_val2017.json" \
--val_im_dir "/home/csprh/scratch/GOT10k/got10k_filtered/images/val2017" \
--start_over \
--fixed_start_frame

python train.py --batch_size 1 \
--run_name "Lwdetr_medium_dyn" \
--patch_size 256 \
--log_path "./log" \
--load_tmt "./tmt_checkpoints/turb_model_best_dyn.pth" \
--train_path "/home/csprh/scratch/COMBO_COCO_DATA/STATIC_GT_P2S_MP4S/traindyn" \
--val_path "/home/csprh/scratch/COMBO_COCO_DATA/STATIC_GT_P2S_MP4S/valdyn" \
--train_ann_file "/home/csprh/scratch/GOT10k/got10k_filtered/annotations/instances_train2017.json" \
--train_im_dir "/home/csprh/scratch/GOT10k/got10k_filtered/images/train2017" \
--val_ann_file "/home/csprh/scratch/GOT10k/got10k_filtered/annotations/instances_val2017.json" \
--val_im_dir "/home/csprh/scratch/GOT10k/got10k_filtered/images/val2017" \
--lwdetr_model "detr_medium" \
--lwdetr \
--fixed_start_frame \
--start_over
```

These are the values (together with the default command line option values) used for the "end to end" comparison results in the accompanying report.

If there is a CUDA out of memory error, reduce the patch size.

### Training Logs

There is a directory is created within the defined log directory for each training run.  This directory contains a subdirectory of checkpoints, a subdirectory of validation images and a text log file (recording.log).
The output images contain three concatenated outputs [the input distorted image (on the left), the mitigated output image (centre) and the ground truth image (on the right)].  

## Testing
Evaluation is achieved using the same script (train.py) as training, but with the --eval flag

```bash
python train.py --batch_size 1 \
--run_name "detr_medium_static" \
--patch_size 256 \
--train_path "./dataset_example/DYNAMIC_1/train2017carperson" \
--val_path "./dataset_example/DYNAMIC_1/val2017carperson" \
--log_path "./log" \
--train_ann_file "./dataset_example/coco_processed/annotations/instances_train2017_crop_carperson.json" \
--train_im_dir ./dataset_example/coco_processed/images/train2017carperson \
--val_ann_file "./dataset_example/coco_processed/annotations/instances_val2017_crop_carperson.json" \
--val_im_dir "./dataset_example/coco_processed/images/val2017carperson" \
--load_tmt "./tmt_checkpoints/turb_model_best.pth" \
--start_over  \
--eval

python train.py --batch_size 1 \
--run_name "Lwdetr_medium_static" \
--patch_size 256 \
--train_path "./dataset_example/DYNAMIC_1/train2017carperson" \
--val_path "./dataset_example/DYNAMIC_1/val2017carperson" \
--log_path "./log" \
--train_ann_file "./dataset_example/coco_processed/annotations/instances_train2017_crop_carperson.json" \
--train_im_dir ./dataset_example/coco_processed/images/train2017carperson \
--val_ann_file "./dataset_example/coco_processed/annotations/instances_val2017_crop_carperson.json" \
--val_im_dir "./dataset_example/coco_processed/images/val2017carperson" \
--load_tmt "./tmt_checkpoints/turb_model_best.pth" \
--lwdetr \
--lwdetr_model "detr_medium" \
--start_over \
--eval

```

### Training Options
| Query Flags | Description |
|:------------|-------------|
| ```--epochs``` | Number of epochs to train |
| ```--batch_size``` | batch size |
| ```--lwdetr_model``` | choose lwdetr model: detr_tiny, detr_small, detr_medium, detr_medium, detr_xl |
| ```--lwdetr``` | boolean, true if you want to use lwdetr over detr |
| ```--losses``` | how to calculate loss: choose from alternate, justtmt, justdetr, weighted |
| ```--tmt_loss_weight``` | the weight when using losses = weighted |
| ```--patch_size``` | patch size (in pixels: both width and height).  Decrease if there are CUDA memory issues |
| ```--detr_frame``` | which frame to choose for object detection |
| ```--detr_frame``` | which frame to choose for object detection |
| ```--log_frequency``` | frequency (in iterations) that the metrics are logged to the log |
| ```--val_period``` | number of iterations between validation tests.  Validation tests the validation dataset and updates the best model if the performance (in terms of PSNR) |
| ```--num_frames``` | number of contiguous frames to be input and output in the model |
| ```--img_out_frequency``` | frequency (in iterations) that the output images are output to the ./log/imgs directory |
| ```--learning_rate``` | initial learning rate for the optimiser (for TMT)|
| ```--resume_detr``` | load DETR model weights from checkpoint |
| ```--run_name``` | name of the run (for log labelling) |
| ```--model_save_frequency``` | number of iterations to save checkpoint |
| ```--num_workers``` | number of workers used for dataloader |
| ```--num_classes``` | number of detection classes (set to 91 for the usually loaded DETR model)|
| ```--load_tmt``` | if this exists then load a model from this path (i.e. place where a previously trained .pth model is found) |
| ```--log_path``` | path to save logging files and images |
| ```--train_ann_file``` | path to json file showing coco data annotations |
| ```--val_ann_file``` | path to json file showing coco data annotations |
| ```--train_im_dir``` | train_im_dir is where the coco images are stored for training|
| ```--val_im_dir``` | val_im_dir is where the coco images are stored for validation|
| ```--train_path``` | path to the training data (formatted as per TMT D3 project) |
| ```--stages``` | one or two stage method (only used for metrics in training) |
| ```--start_over``` | if model is loaded then is start_over is defined then the iterations and epochs are both reset to zero (rather than loaded from the saved model)|
| ```--mambadef``` | use mamba3d transform for deblur network|
| ```--swindef``` | use swin3d transform for deblur network|

The other training options are from the DETR project https://github.com/facebookresearch/detr


## Main Script Description

* ```train.py``` This script trains a one or two stage model for dynamic content using Swin3D components within a TMT structure 


## Directory Structure (of code directory)
```
code
 ┣ FD                           Fractal Dimension Loss / Regularisation
 ┣ data
 ┃ ┗ dataset_video_train.py     Dataloader code etc.
 ┣ dataset_example              Directory containing input data
 ┣ model
 ┃ ┣ detr                       DETR object detection code
 ┃ ┣ lwdetr                     LWDETR object detection code
 ┃ ┣ complex_mamba_triton.py    Triton accelerated inner mamba code (not currently used)
 ┃ ┣ nnMambaVidX.py             Complex Mamba Unet (not currently used)
 ┃ ┣ ssm_x_opt.py               Inner Complex Mamba code (not currently used)
 ┃ ┣ DefConv.py                 Deformable Convolution code
 ┃ ┣ nnMamba.py                 Mamba code
 ┃ ┣ SwinUnet_3D.py             Swin based UNet code (3D)
 ┃ ┗ TMT_DC.py                  Main model source code
 ┣ util                         Utilities from DETR
 ┣ utils
 ┃ ┣ general.py                 General utilities
 ┃ ┣ losses.py                  Loss code
 ┃ ┣ scheduler.py               Scheuler code
 ┃ ┗ utils_image.py             Image based utils
 ┣ train.py                     Training code
 ┣ tmt_checkpoints              Pre trained TMT based checkpoints
 ┣ VideoMamba                   Video Mamba
 ┣ Deformable-DETR              Deformable-DETR
 ┣ dockerfile                   Docker definition
 ┗ README.md                    This file
```



## Bibtex
```
@inproceedings{DMAT_WACV2026,
  title     = {{DMAT}: An End-to-End Framework for Joint Atmospheric Turbulence Mitigation and Object Detection},
  author    = {Paul Hill and Zhiming Liu and Alin Achim and David Bull and Nantheera Anantrasirichai},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026},
}
```
