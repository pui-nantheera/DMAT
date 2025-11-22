# DMAT: An End-to-End Framework for Joint Atmospheric Turbulence Mitigation and Object Detection [WACV2026]

[[Paper](https://arxiv.org/pdf/2507.04323)] [[Real dataset](https://zenodo.org/records/13737763)] [[Synthetic datasets](https://zenodo.org/records/17673509)] [[Pretrained models](https://zenodo.org/records/17673509)]

## Description

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
sudo docker run -v ./code:/app/code -it --gpus all turb_mit bash 
cd code
# Now you can run any of the scripts below
```

## Training

```bash
python train.py --batch_size 1 --patch_size 128 --train_path "./dataset_example/train" --val_path "./dataset_example/train" --log_path "./log"
```

These are the values (together with the default command line option values) used for the "end to end" comparison results in the accompanying report.

If there is a CUDA out of memory error, reduce the patch size.

### Training Logs

There is a directory is created within the defined log directory for each training run.  This directory contains a subdirectory of checkpoints, a subdirectory of validation images and a text log file (recording.log).
The output images contain three concatenated outputs [the input distorted image (on the left), the mitigated output image (centre) and the ground truth image (on the right)].  

## Testing

```bash
python test.py --input_path /home/csprh/Paul/datasets/datasets/TMT/van/Test/turb/van.mp4 --out_path ./out.mp4 --model_path /home/csprh/Paul/PROTOCODE/TMT_AFFINE/Atmosphere-Turbulence-Mitigation/code/log/TMT_default_mamba_07-30-2024-09-29-24/checkpoints/latest.pth --save_video


python test.py --input_path /home/csprh/Paul/datasets/datasets/TMT/van/Test/turb/van.mp4 --out_path ./out2.mp4 --model_path /mnt/d/Paul/models/model_best.pth --save_video

python test_folder.py --input_folder /home/csprh/Paul/datasets/TMT/COCO/DYNAMIC_1/turb --out_folder /home/csprh/Paul/datasets/TMT/COCO/RESULTS_1 --model_path /home/csprh/Paul/PROTOCODE/TMT_AFFINE/Atmosphere-Turbulence-Mitigation/code/log/TMT_default_mamba_07-30-2024-09-29-24/checkpoints/model_100000.pth

```

These are the values (together with the default command line option values) used for the "end to end" comparison results in the accompanying report.

#### Note
For the `out_path`, please remember to contain the output file name you want, for example: `'code/log/out.mp4'` 

### Training Options
| Query Flags | Description |
|:------------|-------------|
| ```--epochs``` | Number of epochs to train |
| ```--batch_size``` | batch size |
| ```--patch_size``` | patch size (in pixels: both width and height).  Decrease if there are CUDA memory issues |
| ```--log_frequency``` | frequency (in iterations) that the metrics are logged to the log |
| ```--val_period``` | number of iterations between validation tests.  Validation tests the validation dataset and updates the best model if the performance (in terms of PSNR) |
| ```--num_frames``` | number of contiguous frames to be input and output in the model |
| ```--img_out_frequency``` | frequency (in iterations) that the output images are output to the ./log/imgs directory |
| ```--learning_rate``` | initial learning rate for the optimiser |
| ```--run_name``` | name of the run (for log labelling) |
| ```--model_save_frequency``` | number of iterations to save checkpoint |
| ```--num_workers``` | number of workers used for dataloader |
| ```--load``` | if this exists then load a model from this path (i.e. place where a previously trained .pth model is found) |
| ```--log_path``` | path to save logging files and images |
| ```--train_path``` | path to the training data (formatted as below) |
| ```--stages``` | one or two stage method (only used for metrics in training) |
| ```--start_over``` | if model is loaded then is start_over is defined then the iterations and epochs are both reset to zero (rather than loaded from the saved model)|
| ```--mambadef``` | use mamba3d transform for deblur network|
| ```--swindef``` | use swin3d transform for deblur network|

### Testing Options
| Query Flags | Description |
|:------------|-------------|
| ```--total_frames``` | total number of frames to process |
| ```--start_frame``` | first frame to process |
| ```--model_path``` | model path to load (i.e. place where a previously trained .pth model is found) |
| ```--input_path``` | path of input video |
| ```--out_path``` | path of output|
| ```--save_images``` | if true then save images |
| ```--save_videos``` | if true then save videos |
| ```--resize_ratio``` | spatial resize ratio for input and output images / video (both width and height) |
| ```--concatenate_input``` | output the input and result images concatenated side by side |
| ```--mambadef``` | use mamba3d transform for deblur network|
| ```--swindef``` | use swin3d transform for deblur network|

## Main Script Description

* ```train.py``` This script trains a one or two stage model for dynamic content using Swin3D components within a TMT structure 
* ```test.py``` This script implements a full video inference (using spatio-temporal patches)  of a trained model

## Directory Structure (of code directory)
```
code
 ┣ FD                           Fractal Dimension Loss / Regularisation
 ┣ data
 ┃ ┗ dataset_video_train.py     Dataloader code etc.
 ┣ dataset_example
 ┣ model
 ┃ ┣ DefConv.py                 Deformable Convolution code
 ┃ ┣ SwinUnet_3D.py             Swin based UNet code (3D)
 ┃ ┗ TMT_DC.py                  Main model source code
 ┣ utils
 ┃ ┣ general.py                 General utilities
 ┃ ┣ losses.py                  Loss code
 ┃ ┣ scheduler.py               Scheuler code
 ┃ ┗ utils_image.py             Image based utils
 ┣ test.py                      Testing code
 ┣ train.py                     Training code
 ┣ dockerfile                   Docker definition
 ┣ pyproject.toml               Poetry definition (currently not supported)
 ┗ README.md                    This file
```
## Input Data Structure

When pointing to a training dataset it must be formatted as follows
```
.
├── gt
└── turb
```
Where each directory contains .mp4 video sequences for each training video.  gt is the ground truth (non distorted), turb is the distorted version.  If the ground truth is a single static image, then the image is just repeated with the appropriate gt .mp4 video for the same length as the output video.

