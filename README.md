# PredictingMaskAndDepthUsingDeepLearning
Predicting Mask and Depth of image from background and foreground superimposed on background Using Deep Learning Techinques

# I. Problem Statement

In this project background image and foreground image superimposed on background image are given, task is predict

1. Mask Corresponding to the foreground
2. Depth map of image

The dataset is given in 10 zip files (batch1_images.zip, batch2_images.zip ..., batch10_images.zip). Each zip contains the following
 folders:

bg_jpg : Background jpg images

fg_bg_jpg : Foreground image superimposed on background image

mask_black_jpg : Ground truth of Mask of foregorund image on black background

depth_fg_bg_jpg: Ground truth of depth image

The complete dataset is available in google drive link:

https://drive.google.com/drive/folders/1YF4HvfTdDwDLYPmBokx4b5QzInMVyAzA?usp=sharing

Jupyter Notebook link:
https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/assignment15_final_api.ipynb


If you face any issues in opening Jupyer Notebook, please use the link:

https://colab.research.google.com/drive/1xylqpDJOHght0IcGGblgmMTSK4OKtLoN?usp=sharing

Major Highlights:

-  Use fully API based implementation 
-  Used UNet architecture with only 148K parameters 
-  Used Image augmentation i. Gaussian Noise ii. ColorJitter
-  IoU value for predicyed mask is close to 0.95, while depth images are very closer to ground truth 
- Used various profiling tools tensorboard, cprofile, GPU profiler 
- Measured time during training, copying, unzipping images
- Only 5 epochs are used to achive the result

## Running the code

For running the code, you can take the Jupyter Notebook assignment15_final_api.ipynb links which I provided above,
make sure you clone the repository inside the notebook to get the API codes

git clone https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning.git 

Alternatively , if you directly want to use the Jupyter Notebook, please take API from the google drive link:

https://drive.google.com/drive/folders/1YTvb7V0eDfn5MZwBbc4msFkWKH5ArotI?usp=sharing 


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
- Model is used to foreward pass through the neural network and predicts mask 
- Loss Function calculates loss value between predicted mask image and ground truth mask image as well 
as predicted depth image and ground truth depth image. The loss value is backpropagated through the neural network
and weights of model are updated


Outputs are:

- 160x160 mask of foreground on black background
- 80x80 predicted depth image

The code for Traing is available in URL:

 https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/utils/train_test_util.py
 
 (Please look for train method)


## i. Data Loader

Datal Loader performs loading of data from the images.


The workflow for dataloader is explained in the flowchart below:

![Project Report](/doc_images/data_loader_workflow.png)

The process involved:
, 
- Copy all the zip files from the google drive to the Google colab local folder /content/data
- Unzip each of zip in a respective batch folder. For example batch1_images.zip is unzipped to /content/data/batch1 folder.

Similar process is done for other batches as well
- There are two datasets:
  i. TrainImageDataset - This dataset is constructed from 9 zip files (batch1_images.zip, batch1_images.zip, ... batch9_images.zip) unzipped in
respetive batch folder (batch1, batch2 4, .. batch9)  and used for training.
  ii. TestImageDataset - This dataset is constructed using only batch10_images.zip unzipped in batch10 folder

Records are populated as below
 - Multi level index (batch id, offset) of all the files in fg_bg_jpg folder
  - The  __getitem__ method takes index as an argument.
 - index is used to calculate batch_id by dividing index by 40000. Remainder is used to calculate offset
 - Once the fg_bg image file is identified the corresponding background image file is identified based on naming convention.
 For exmaple of fg_bg image file name is fg_bg_1_100_1_15.jpg then by convention second number after fg_bg will be background
 image, in this case it will be bg_100.jpg and it will be avaialble in bg_jpg folder under respective batch id directory
 
 Based on convention the ground truth mask image, filename will have same suffix as the fg_bg image file name. For example if fg_bg image
 filename is fg_bg_1_100_1_15.jpg, file name correspoding to ground truth mask image will be 
 bg_mask_1_100_1_15.jpg, which will be available in mask_black_jpg folder under the batch id directory
 
 Similary, ground truth  depth image filename will have the same suffix as the fg_bg image file name. For example, if fg_bg image
 filename is fg_bg_1_100_1_15.jpg, the  filename correspoding depth image will be 
 depth_1_100_1_15.jpg, which will be available in depth_fg_bg_jpg directory under the respective batch directory
 
 The code for dataloader is available in URL:
 
 https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/data_loaders/fg_bg_images_data_loader.py
  

## ii. Data Augmentation

For the training images there are two augmentations used:

- ColorJitter from torchvision with brightness:0.075, contrast:0.075, saturation:0.075, hue:0.075
- Custom class GaussianNoise with mean 0, standard deviation: 0.05 with probability: 0.2. 

Same Data augmentations are applied on input bg, fg_bg, mask, as well as ground truth mask and depth images


The code for data augmentation is available in URL: 
 https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/data_transformations/fg_bg_images_data_transformation.py


## iii. Model

I have used UNet Model for this. UNet Model is suitable for segmentation works.

Original UNet architecture is as below [Source: https://towardsdatascience.com/u-net-b229b32b4a71]

![Project Report](/doc_images/unet_architecture.png)

The architecture looks like a ‘U’ which justifies its name. This architecture consists of three sections: The contraction, 
The bottleneck, and the expansion section. The contraction section is made of many contraction blocks. Each block takes an input 
applies two 3X3 convolution layers followed by a 2X2 max pooling. The number of kernels or feature maps after each block doubles
 so that architecture can learn the complex structures effectively. The bottommost layer mediates between the contraction layer 
 and the expansion layer. It uses two 3X3 CNN layers followed by 2X2 up convolution layer.
 
But the heart of this architecture lies in the expansion section. Similar to contraction layer, it also consists of several 
expansion blocks. Each block passes the input to two 3X3 CNN layers followed by a 2X2 upsampling layer. Also after each block 
number of feature maps used by convolutional layer get half to maintain symmetry. However, every time the input is also get appended 
by feature maps of the corresponding contraction layer. This action would ensure that the features that are learned while contracting
 the image will be used to reconstruct it. The number of expansion blocks is as same as the number of contraction block. After that,
 the resultant mapping passes through another 3X3 CNN layer with the number of feature maps equal to the number of segments desired.


Original UNet Model has around 25 Million parameters, so to reduce parameters:

- Used Depthwise Separable Convolution
- Number of channels are also reduced

After reduction total number of parameters is 748K, which is less than even 1 Million.

The output of the model has two heads one for predicting mask of size (160x160) and another head is for predicting
depth (80x80)

The code for Model is available in URL:

https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/models/unet_model_small.py


Model weights are saved after every 100 batches are processed so that I can recover in case any crash happens.
It also helps to select best model based on results
 All the model weights are available in URL:
 
 
https://drive.google.com/drive/folders/1--BfoDLcQeCW3dvh2_52zMf4no6pg-YJ?usp=sharing





## iv. Loss Function

There are two loss functions used:

### 1. BCEWitLogitsLoss : 

This loss combines a `Sigmoid` layer and the Binary Cross Entropy Loss in one single class.

I have used this loss function for mask predictions as it requires pixel level comparison

### 2. SSIM : 

The structural similarity (SSIM) index is a method for predicting the perceived quality of digital television and cinematic pictures,
 as well as other kinds of digital images and videos. SSIM is used for measuring the similarity between two images.
 
 I have used SSIM as loss function for calculating loss between predicted and ground truth depth images.
 
 
 Total Loss I have used the formula:
 
 Total Loss = 2 * ( BCEWitLogitsLoss for mask images) + (1 - SSIM loss for depth images)
 
 ( I have used 1-SSIM Loss as SSIM Loss is in the range 0 to 1 with 1 being perfect match)
 
 SSIM is implemented in https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/utils/ssim_util.py. 
 The code is taken from https://github.com/Po-Hsun-Su/pytorch-ssim
 
 
## v. Metric Used

Loss : This calculates total loss value for ( predicted mask image with ground truth mask image) and (predicted
depth image with ground truth depth image)

IoU: IOU for predicted mask is calculated using jaccard similarity cofficient. Jaccard similarity coefficient, defined as the
 size of the intersection divided by the size of the union of two label sets, is used to compare set of predicted labels for a
 sample to the corresponding set of labels in y_true 

First for each pixel vlaue  predicted mask image and ground mask image a threshold is defined. If pixel value is more than threshold
then it is considered 1 else zero. Next jaccard_score of sklearn.metrics is used to calculate IoU value.

The code for calculating IoU is:
 https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/utils/iou_util.py


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

The following diagram shows the main components of training

![Project Report](/doc_images/testing_components.png)


Here inputs are
- 160x160 background images
- 160x160 foreground images superimposed on background


Processing are done by:

- Data loader loads the data. Dataloader uses the images from batch10 i.e. batch10_images.jpg
- Model is used to foreward pass through the neural network and predicts mask  


Outputs are:

- 160x160 mask of foreground on black background
- 80x80 predicted depth image


For testing, best model from training is loaded and then evaluation is done on the input images

The code for Testing is available in URL:

 https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/utils/train_test_util.py
 
 (Please look for test method)



# IV. Results

All the results are available in Jupyter Notebook 
https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/assignment15_final_api.ipynb

(Training results are in training cell output titled "Perform training of the model and periodically display statistics, mask , depth images generated, IoU value"
, and testing results are in testing cell output titled "Load the best model and perform test and Test dataset and show results")

## Results from Testing:


## i. 

### Background Image:
![test_bg_1](/results/test_bg_1.jpg)

### Foreground and Background Image:

![test_fg_bg_1](/results/test_fg_bg_1.jpg)

### Ground Truth Mask Image:

![test_fg_bg_1](/results/test_ground_truth_mask_1.jpg)

### Predicted Mask Image:

![test_fg_bg_1](/results/test_predicted_mask_1.jpg)


Predicted Mask IoU value :  0.9565858394563042

### Ground Truth Depth Image (with plasma display):

![test_fg_bg_1](/results/test_ground_truth_depth_1.jpg)

### Predicted Mask Image (with plasma display):

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


Predicted Mask IoU value :  0.9478972876254577

### Ground Truth Depth Image (with plasma display):

![test_fg_bg_2](/results/test_ground_truth_depth_2.jpg)

### Predicted Depth Image (with plasma display):

![test_predicted_depth_2](/results/test_predicted_depth_2.jpg)


## Result from Training

## i. 

### Background Image:
![test_bg_1](/results/train_bg_1.jpg)

### Foreground and Background Image:

![test_fg_bg_1](/results/train_fg_bg_1.jpg)


### Predicted Mask Image:

![test_fg_bg_1](/results/train_mask_1.jpg)

Mask IoU vlaue: 0.9422367689214051


### Predicted Depth Image (with plasma display):

![test_predicted_depth_1](/results/train_depth_1.jpg)




# V. Profiling:


## i. Tensorboard

TensorBoard is a profiler and visualization tool, it is  used for the following purposes:
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

![loss_plot](/tensor_board_plots/loss_plot.png)


### Analysis: 

From the plot it is evident that training loss reduces in very first epoch to around 0.2 and stays flat there

The testing plot is flucutaing in a small range between 0.118 and 0.126. So loss value in test is better than training. 
This may be because I am using image augmentation during training


### Training and Testing IoU vlaues

![iou_plot](/tensor_board_plots/iou_plot.png)

From the training IoU it is evident that IoU values increases initially to around 0.92 and then it remains steady between
0.94 and 0.95

For the testing, IoU value remains steady between small range of 0.948 and 0.95


Tensor board captureed profiles files is available in URL:

https://drive.google.com/file/d/1Eye2F9UmXo_uLTueZjGGNbFiVhbxgyB8/view?usp=sharing


## ii. cProfile:

cProfile is used  for profiling Python programs. 

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

The full raw stats generated from cprofile is available in URL:


https://drive.google.com/file/d/12SyJc8aK_wlmXU2fI-_JG2cSOaW_dgdD/view?usp=sharing


## Analysis:

The train_test_utils.py line numbers 77, 21, 24 and 29 consume lot of time. Line no 77 is related to train method, line number 24 and 29
is related to GPU profiling hooks used, which can be removed 

Another component which is consuming time is unet_model_small.py forward function in line 115 also consumes lot of time

Python libraries like tornado/stack_context.py zmq/eventloop/zmq_stream.py also consumes lot of time

Another time consuming method is tqdm/notebook.py tqdm/std.py, we can explore any ligter version available

torch/util/data/data_loader.py and torch/util/data/_utils/fetch.py also consumes time, which can be improved by 
increasing num_workers attribute.

cProfile stats file is available in URL:

https://drive.google.com/file/d/12SyJc8aK_wlmXU2fI-_JG2cSOaW_dgdD/view?usp=sharing




## iii. GPU Profiling


I have instrumented the training code for GPU profiling based on article:

 https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks
 
 The Plots did not work. But I have converted teh memory profile to pandas dataframe and from that I have downaloaded as CSV.
 
 A few records are as below:

| layer_idx | call_idx |  layer_type              |  exp   | hook_type |    mem_all   |  mem_cached     |
| ----------|----------|--------------------------|--------| ----------|--------------|---------------- |
|    0      |    0     |    UNet                  | exp_0  |   pre     |   35588096   |   341835776     |
|    1      |    1     |  DoubleConv              | exp_0  |   pre     |   56559616   |   341835776     |
|    2      |    2     |   Sequential             | exp_0  |   pre     |   56559616   |   341835776     |
|    3      |    3     | depthwise_separable_conv | exp_0  |   pre     |   35588096   |   341835776     |
|    4      |    4     |    Conv2d                | exp_0  |   pre     |   56559616   |   341835776     |
|    4      |    5     |    Conv2d                | exp_0  |   fwd     |   77531136   |   341835776     |
|    5      |    6     |    conv2d                | exp_0  |   pre     |   77531136   |   341835776     |
|    5      |    7     |    Conv2d                | exp_0  |   pre     |   405211136  |   1000341504    |
|    3      |    8     | depthwise_separable_conv | exp_0  |   fwd     |   405211136  |   1000341504    |
|    6      |    9     |    BatchNorm2d           | exp_0  |   pre     |   405211136  |   1000341504    |

Last few records depthwise_separable_conv, BatchNorm2d are using mem_cached value of  1000341504 which is too high, needs
further attention.

The code for GPU profiling is available in URL:

https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/utils/train_test_util.py

(please look for methods _get_gpu_mem, _generate_mem_hook, _add_memory_hooks, log_mem)

Full GPU Profiling gpu_profiler_stats.csv file is available in:

https://drive.google.com/file/d/1Es4bPPQIa2937jxwIOWNi4ztVIf9zVtM/view?usp=sharing

 

## iv. Time Measurements

I have measured time copyinng zip files, unzipping the zip file, as well as training times.
 
(All measurements units are seconds)

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


### a. Epoch vs Training time Plot

![doc_images](/doc_images/epoch_vs_training_time.png)

### b. Epoch vs Data Loading time Plot

![doc_images](/doc_images/epoch_vs_dataloading_time.png)

### b. Epoch vs Misc time Plot

![doc_images](/doc_images/epoch_vs_misc_time.png)


## Analysis: 

From this data training time gradually increases upto around 60 seconds then sligtly decreases. Data Loading time and
Misc Time are almost constant across epochs.

Data loading time can be further reduced by changing num_workers attribute in Dataloader. 

## vi. MACs value for Model:

multiply-and-accumulate (MAC) operations gives how  better model will perform in terms of number of operations. Installed thop library
and used profile method to calculated MACs value as below:

MACs:  892294400.0

Code for calculating MACs value is in  the Jupyter Notebook:

https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/assignment15_final_api.ipynb

(Please look for cell under the title "Calculate and display MACS value of the model")


# VI. Code Structure

|  Path                                                     |        Comment                                                          |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
|  assignment15_final_api.ipynb                             |  Main Jupyter Notebook for training and testing the model               |
|  data_loaders/fg_bg_images_data_loader.py                 |  Image Data Loader                                                      |
|  data_transformations/fg_bg_images_data_transformation.py |  Transformation and Augmentation                                        | 
|  utils/iou_util.py                                        |  Utility for calculating IoU                                            |
|  utils/ssim_util.py                                       |  Utility for SSIM Loss calculation                                      |
|  utils/plot_util.py                                       |  Utility for plotting images                                            |      
|  utils/train_test_util.py                                 |  Utility for training testing and GPU profiling                         |
|  results                                                  |  Image Results are stored                                               |
|  cprofile_plots                                           |  Cprofile plots are stored                                              |
|  tensorboard_plots                                        |  Tesnsorboard plots are stored                                             |

 


# VII. Problems Faced and how I addressed

## i. Loss Functions
Initilaly I was using BCELogitsLoss for both mask and depth predictions but depth image quality was not good. When I used
SSIM for depth and BCEWitLogitsLoss for Mask, I found the depth images were better and even mask images IoU is around 0.95 which 
is quite good

## ii. Unzip the batch zip image files

Initially I was trying to unzip the images zip file in google drive itself. Each batch of images were taking close to 
2 hours. So, I changed the strategy and tried copying the image zip files locally to Colab and then unzipped, it significantly
reduces the time. However, the downside is that 


## iii. Colab GPU access disabled:
My Colab account for GPU access was suspended giving reason that my usage limit is high. So changed to Colab Pro
by paying monthly subscription

## iv. Jupyter Notebook hung and colab disconnected frequently
The Jupyter notebook got hung lot of times (after say internet got disconneted for sometime). I realized that this is because I was
displaying too many imges after few iterations. So I reduced number of times image display per epoch to only two times. Even then
I faced issues, so I used the chrome extension as mentioned in telegram group post https://github.com/satyajitghana/colab-keepalive .
As I am also saving model periodically to google drive. In case I can not view Jupyter Notebook, I keep on viewing new model weights 
files are generated to know that Colab is still working on Jupyter notebook

## vi. Batch size selection
Initially I was trying bigger batch size (256,124) but I was getting out of memory. Finally I found that batch size of 100 works
without any memory issue



# VIII. Future works

## i. 
The ground truth depth images were not very high quality. My model would have given better results if ground truths are good quality.
So, as a future I will try good models to generate ground truth depth images

## ii. 
Further analysis of GPU memory profiling can be done

### iii.

 I have to run around 7.5 hours to achieve the result. As Google TPU are faster, I would like to try with TPU
 
### iv.

Apply knowledge acquired different applications like lung cancer detetion etc.



# IX. Conclusion


In this project, I have worked on predicting Mask and Depth of given background and foregroud superimposed background images. 
I have used reduced UNet model of only 748K parameters (i.e. less than 1M parameters) and predicted mask and depth almost
 closer to the ground truth values. Mask IoU is around 0.95.

I have used various profiling tools: tensorboard, cprofile, GPU profiler as well as calculated MACS value for the model.

This project is a great learning opportunity for me.













