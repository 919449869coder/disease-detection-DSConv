# DS-Conv

This project aims to utilize CycleGAN for image simulation and DSConv for training a target detection model. Below are the detailed instructions and requirements for successful execution.

## Table of Contents

- [Step 1: Clone and Set Up CycleGAN](#step-1-clone-and-set-up-cyclegan)
- [Step 2: Download and Prepare Datasets](#step-2-download-and-prepare-datasets)
- [Step 3: Download Pre-trained CycleGAN Model](#step-3-download-pre-trained-cyclegan-model)
- [Step 4: Image Simulation](#step-4-image-simulation)
- [Step 5: Train CycleGAN Model](#step-5-train-cyclegan-model)
- [Step 6: Test CycleGAN Model](#step-6-test-cyclegan-model)
- [Step 7: Use CycleGAN for Disease Image Generation](#step-7-use-cyclegan-for-disease-image-generation)
- [Requirements](#requirements)
- [Usage Instructions](#usage-instructions)
- [Examples](#examples)
- [Common Issues](#common-issues)
- [License](#license)
#### cyclegan ####

## Step 1: Clone and Set Up CycleGAN

1. Clone the CycleGAN repository:
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
2.Navigate to the CycleGAN directory：
cd pytorch-CycleGAN-and-pix2pix
## Step 2: Prepare Datasets
bash ./datasets/....
## Step 3:  Download Pre-trained CycleGAN Model
bash ./scripts/download_cyclegan_model.sh horse2zebra
The pre-trained model is saved at ./checkpoints/{name}_pretrained/latest_net_G.pth.
## Step 4: Image Simulation
python simulate_images.py --input_dir path/to/input --output_dir path/to/output
## Step 5: Train 
bash ./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
python -m visdom.server
## Step 6: Test
bash ./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
## Step 7: Use CycleGAN for Disease Image Generation
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout


#### yolo-dsconv ####
#score_data.yaml
## Step 8: YOLO parameters set
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
train: ./dataset/score/images/train # train images
val: ./dataset/score/images/val # val images
#test: ./dataset/score/images/test # test images (optional)

# Classes
names:
  0: NI
  1: LI
  2: HI
## Step 9: YOLO train
yolo task=detect mode=train model=yolov8s.yaml  data=yolov8-C2f-DySnakeConv.yaml epochs=100 batch=64 imgsz=640 pretrained=False optimizer=SGD 
## Step 10: DSConv-GAN model application
The application interface branch contains the application interface.zip file, which includes the application interface we developed.



