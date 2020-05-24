# PredictingMaskAndDepthUsingDeepLearning
Predicting Mask and Depth of image from background and foreground superimposed on background Using Deep Learning Techinques

# I. Problem Statement

In this project backgorund image and foreground image is given, task is predict

1. Mask Corresponding to the foreground
2. Esumation of depth of image

The dataset is given in 10 zip files (batch1_images.zip, batch2_images.zip ..., batch10_images.zip). Each zip contains the following folders:

bg_jpg : Background jpg images
fg_bg_jpg : Foreground image superimposed on background image
mask_black_jpg : Ground truth of Mask of foregorund image on black background
depth_fg_bg_jpg: Ground truth of depth image

The complete dataset is available in google drive link:

https://drive.google.com/drive/folders/1YF4HvfTdDwDLYPmBokx4b5QzInMVyAzA?usp=sharing


# II. Training

The following diagram shows the main components of training

![Project Report](/doc_images/training_components.png)


Here inputs are
- 160x160 background images
- 160x160 foreground images superimposed on background
- 160x160 ground truth of mask of foreground on black background
- 80x80 ground truth depth image

Processing are done by:

- Data loader loads the data, while image augmentation performs augmentation of image. 
- Model is used to foreward pass through the neural network and predicts mask and . 
- Loss Function calculates loss value between ground truth of  predicted mask and ground truth mask as well 
as predicted depth and ground truth depth and the loss value is backpropagated through the neural network
and weights of model are updated


Outputs are:

- 160x160 mask of foreground on black background
- 80x80 predicted depth image


## i. Data Loader

Datal Loader performs loading oda data from the images.

The process involved:
, 
- Copy all the zip files from the google drive to the Google colab local folder /content/data
- Unzip each of zip in a respective batch folder. For exampe batch1_images.zip is unzipped to /content/data/batch1 folder.

Similar process is done for other batches as well
- There are two datasets:
  i. TrainImageDataset - This dataset is constructed from 9 zip files (batch1_images.zip, batch1_images.zip, ... batch9_images.zip) unzipped in
respetive batch folder (batch1, batch2 4, .. batch9)  and used for training.
  ii. TestImageDataset - This dataset is constructed using only batch10_images.zip unzipped in batch10 folder

Records are populated as below
 - Multi level index (batch id, offset) of all the files in fg_bg_jpg folder
  - The  __getitem__ method takes index as an argument.
 - index is used to calculate batch_id by dividing batch_id by 40000. Remainder is used to calculate offset
 - Once the fg_bg image file is identied the corresponding background image file is identified based on naming convention.
 For exmaple of fg_bg image file name is fg_bg_1_100_1_15.jpg then by convention second number after fg_bg will be background
 image, in this case it will be bg_100.jpg and it will be avaialble in bg_jpg folder under respective batch id directory
 
 Based on convention the mask image filename will have same suffix so,  fg_bg_1_100_1_15.jpg file correspoding mask image will be 
 bg_mask_1_100_1_15.jpg will be available in mask_black_jpg folder under the batch id directory
 
 Similary, depth image filename will have the same suffix so fg_bg_1_100_1_15.jpg fg_bg_1_100_1_15.jpg file correspoding mask image will be 
 depth_1_100_1_15.jpg will be available in mask_black_jpg directory under the respective batch directory
 

## ii. Data Augmentation

For the training images there are two augmentations used:

- ColorJitter from torchvisiong with brightness:0.075, contrast:0.075, saturation:0.075, hue:0.075
- Custom class GaussianNoise with mean 0, standard deviation: 0.05 with probability: 0.2. 

Same Data augmentations are applied on input bg, fg_bg, mask, depth images

## iii. Model

I have used UNet Model for this. UNet Model is suitable for segmentation work.

Original UNet Model has around 25 Million parameters, so reduced parameters:

- Used Depthwise Separable Convolution
- Number of channels are also reduced

After reduction total number of parameters is 748K, which is less than even 1 Million.

The output of the model has two heads one for predicting mask of size (160x160) and another head is for predicting
depth (80x80)

## iv. Loss Function

There are two loss functions used:

BCEWitLogitsLoss : This loss combines a `Sigmoid` layer and the Binary Cross Entropy Loss in one single class.

- I have used this loss function for mask predictions as it requires pixel level comparison

SSIM : The structural similarity (SSIM) index is a method for predicting the perceived quality of digital television and cinematic pictures, as well as other kinds
 of digital images and videos. SSIM is used for measuring the similarity between two images.
 
 I have used SSIM as loss function for calculating loss between predicted and ground truth depth images
 
 
 Total Loss I have used the formula:
 
 Total Loss = 2 * ( BCEWitLogitsLoss for mask images) + (1 - SSIM loss for depth images)
 
 
## v. Metric Used

Loss : This calculates total loss value for ( predicted mask image with ground truth mask image) and (predicted
depth image with ground truth depth image)

IoU: IOU for predicted mask is calculated using jaccard similarity cofficient. Jaccard similarity coefficient, defined as the size of the intersection 
divided by the size of the union of two label sets, is used to compare set of predicted labels for a sample to the corresponding set of
 labels in y_true 

First for each pixel vlaue  predicted mask image and ground mask image a threshold is defined. If pixel value is more than threshold
then it is considered 1 else zero. Next jaccard_score of sklearn.metrics is used to calculate IoU value


## v. Hyper parameters

The following Hyperparameters are used:

Batch Size : 100
Number of Epochs : 5
Initial Learning Rate: 0.01
Momentum : 0.9
Weight Decay: 1e-5


## Optimizer and Scheduler Used:

Used SGD Optimizer with momentum. Learning Rates are reduced using StepLR with step size 2 and gamma 0.1



# III. Testing


# IV. Results

A few results from Test are as below:


## i. 

### Background Image:
![test_bg_1](/results/test_bg_1.jpg)

### Foreground and Background Image:

![test_fg_bg_1](/results/test_fg_bg_1.jpg)

### Ground Truth Mask Image:

![test_fg_bg_1](/results/test_ground_truth_mask_1.jpg)

### Predicted Mask Image:

![test_fg_bg_1](/results/test_predicted_mask_1.jpg)

### Ground Truth Mask Image:

![test_fg_bg_1](/results/test_ground_truth_mask_1.jpg)

### Predicted Mask Image:

![test_fg_bg_1](/results/test_predicted_mask_1.jpg)

Predicted Mask IoU value :  0.9565858394563042

### Ground Truth Depth Image:

![test_fg_bg_1](/results/test_ground_truth_depth_1.jpg)

### Predicted Mask Image:

![test_predicted_depth_1](/results/test_predicted_depth_1.jpg)



## ii. 

### Background Image:
![test_bg_2](/results/test_bg_2.jpg)

### Foreground and Background Image:

![test_fg_bg_2](/results/test_fg_bg_2.jpg)

### Ground Truth Mask Image:

![test_fg_bg_2](/results/test_ground_truth_mask_2.jpg)

### Predicted Mask Image:

![test_fg_bg_2](/results/test_predicted_mask_2.jpg)

### Ground Truth Mask Image:

![test_fg_bg_2](/results/test_ground_truth_mask_2.jpg)

### Predicted Mask Image:

![test_fg_bg_2](/results/test_predicted_mask_2.jpg)

Predicted Mask IoU value :  0.9565858394563042

### Ground Truth Depth Image:

![test_fg_bg_2](/results/test_ground_truth_depth_2.jpg)

### Predicted Mask Image:

![test_predicted_depth_2](/results/test_predicted_depth_2.jpg)


Result from Training

## i. 

### Background Image:
![test_bg_1](/results/train_bg_1.jpg)

### Foreground and Background Image:

![test_fg_bg_1](/results/train_fg_bg_1.jpg)


### Predicted Mask Image:

![test_fg_bg_1](/results/train_predicted_mask_1.jpg)

Mask IoU vlaue: 0.9422367689214051


### Predicted Mask Image:

![test_predicted_depth_1](/results/train_predicted_depth_1.jpg)




# V. Profiling:



## i. Tensor Board

TensorBoard is a profiler and visualization tool  and used for following:
- Tracking and visualizing metrics such as loss and accuracy
- Visualizing the model graph (ops and layers)
- Viewing histograms of weights, biases, or other tensors as they change over time

I have used  SummaryWriter from torch.utils.tensorboard  to add scalar values for metrics, IoU, Loss values for both training and testing

The tensorboard profiles are created using runs folder of Google collab. I created a tar.gz from runs folder, downloaded locally.

After unzipping started tensorboard using command:

tensorboard --logdir=runs

The output is available in http://localhost:6006

Various Tensorboard plots are as below:


### Training and Testing Loss

![loss_plot](/tensor_board_plots/loss_plot.jpg)


### Analysis: 

From the plot it is evident that training loss reduces in very first epoch to around 0.2 and stays flat there

The testing plot is flucutaing in a small range between 0.118 and 0.126. So loss value in test is better than training. 
This may be because I am using image augmentation during training


### Training and Testing IoU vlaues

![iou_plot](/tensor_board_plots/iou_plot.jpg)

From the training IoU it is evident that IoU values increases initially to around 0.92 and then it remains steady between
0.94 and 0.95

For the testing, IoU value remains steady between small range of 0.948 and 0.95



## ii. cProfile:

cProfiler is another  for profiling Python programs. 

I have enabled cProfile by using the following lines at the beginning of Jupyter Notebook

pr = cProfile.Profile()
pr.enable()


At the end of Jupyter notebook, I have disabled cProfile and dumped the stats to a file cprofile_stats.txt

I have downloaded the file cprofile_stats.txt locally and used the cprofilev program to analyze

cprofilev -f cprofile_stats.txt

cProfile output available at http://127.0.0.1:4000

The following are some of the screenshots of cProfile:

![cprofile_plot_1](/cprofile_plots/cprofile_plot_1.png)

![cprofile_plot_2](/cprofile_plots/cprofile_plot_2.png)

![cprofile_plot_2](/cprofile_plots/cprofile_plot_2.png)




## iii. GPU Profiling


I have instrumented the training code for GPU profiling based on article:

 https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks
 
 The Plots did not work. But I have converted teh memory profile to pandas dataframe and from that I have downaloaded as CSV.
 
 A few records are as below:

| layer_idx | call_idx |  layer_type              |  exp   | hook_type |    mem_all   |  mem_cached     |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    0      |    0     |    UNet                  | exp_0  |   pre     |   35588096   |   341835776     |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    1      |    1     |  DoubleConv              | exp_0  |   pre     |   56559616   |   341835776     |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    2      |    2     |   Sequential             | exp_0  |   pre     |   56559616   |   341835776     |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    3      |    3     | depthwise_separable_conv | exp_0  |   pre     |   35588096   |   341835776     |
| ----------|----------|--------------------------| -------|-----------|--------------|-----------------|
|    4      |    4     |    Conv2d                | exp_0  |   pre     |   56559616   |   341835776     |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    4      |    5     |    Conv2d                | exp_0  |   fwd     |   77531136   |   341835776     |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    5      |    6     |    conv2d                | exp_0  |   pre     |   77531136   |   341835776     |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    5      |    7     |    Conv2d                | exp_0  |   pre     |   405211136  |   1000341504    |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    3      |    8     | depthwise_separable_conv | exp_0  |   fwd     |   405211136  |   1000341504    |
| ----------|----------|--------------------------|--------| ----------|--------------|-----------------|
|    6      |    9     |    BatchNorm2d           | exp_0  |   pre     |   405211136  |   1000341504    |

Last few records depthwise_separable_conv, BatchNorm2d are using mem_cached value of  1000341504 which is too high, needs
attentions



 

## iv. Time Measurements

I have measured time copyinng zip files, unzipping the zip file, as well as training times. 


Total time taken for copying zip files:  128.2801535129547

Total time taken for unzipping zip files:  155.29554629325867

The training time I have split into three heads:

i. Training time
ii. Data Loading time
iii. Misc time


|   Epoch      |     Training Time     |    Data Loading Time     |     Misc Time                    |
| -------------|-----------------------|--------------------------|----------------------------------|
|     1        |   3600.63835811615    |    10.604703187942505    |    56.787678956985474            |
|     2        |   3622.2514436244965  |    10.590641975402832    |    59.93659019470215             |
|     3        |   3626.0100643634796  |    10.588401794433594    |    60.445101261138916            |
|     4        |   3661.500823497772   |    10.634032964706421    |    61.81038284301758             |
|     5        |   3653.637369155884   |    10.622773170471191    |    57.52307391166687             |

## Analysis: From this data training time has a slight variance within 60 seconds across epochs. Data Loading time and
Misc Time are almost constant across epochs

vi. MACs value for Model:

multiply-and-accumulate (MAC) operations gives how  better model will perform in terms of number of operations. Used thop library
to calculated MACs value as below:

MACS:  892294400.0


# VI. Code Structure

|  Path                                                     |        Comment                                                          |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
|  assignment15_final_api.ipynb                             |                                                                         |    
|-----------------------------------------------------------|-------------------------------------------------------------------------|
|  data_loaders/fg_bg_images_data_loader.py                 |                                                                         |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
|  data_transformations/fg_bg_images_data_transformation.py |                                                                         |           |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
|  utils/iou_util.py                                        |                                                                         |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
|  utils/ssim_util.py                                       |                                                                         |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
|  utils/plot_util.py                                       |                                                                         |           |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
|  utils/train_test_util.py                                 |                                                                         |           |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
|  results                                                  |                                                                         |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
|  cprofile_plots                                           |                                                                         |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
|  tensorboard_plots                                        |                                                                         |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
 
                                                 

# VII. Problems Faced and how I addressed

## i. Colab accessMy Colab account for GPU access was suspended giving reason that my usage limit is high. So changed to Colab Problem
by pahying monthly subscription

ii. The Colab page got hung lot of times (after say internet got disconneted for sometime). I realized that this is because I was
displaying too many imges after few iterations. So I reduced 

iii. Initilaly I was using BCELogitsLoss for both mask and depth predictions but depth image quality was not good. When I used
SSIM for depth and BCEWitLogitsLoss for Mask, I found the depth images were better and even mask images IoU is around 0.95 which 
is quite good

iv. Initially I was trying to unzip the images zip file in google drive itself. Each batch of images were taking close to 
2 hours. So, I changed the strategy and tried copying the image zip files locally to Colab and then unzipped, it significantly
reduces the time. However, the downside is that 


# VIII. Future works

i. The ground truth depth images were not very high quality. My model would have given better results if ground truths are good quality.
So, as a future I will try good models to generate ground truth depth images

ii. Further analysis of GPU memory profiling can be done

ii. I have to run around 7.5 hours to achieve the result. As Google TPU are faster, I would try to use TPU




# IX. Conclusion


In this project, I have worked on predicting Mask and Depth of given background and foregroud superimposed background images. I have used reduced UNet model of only 748K parameters (i.e. less than 1M parameters) and predicted mask and depth almost closer to the ground truth values. Mask IoU is around 0.95.

I have used various profiling tools: tensorboard, cprofile, GPU profiler as well as calculated MACS value for the model













