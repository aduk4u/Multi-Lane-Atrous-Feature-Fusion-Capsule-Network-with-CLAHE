# Multi-Lane-Atrous-Feature-Fusion-Capsule-Network-with-CLAHE
Multi-Lane Atrous Feature Fusion Capsule Network with Contrast Limited Adaptive Histogram Equalization for Brain Tumor Classification from MRI Images

## The is study is an on-going Laboratory project
### This repo is the official implementation of MLAF-CapsNet

## Abstract
Convolutional neural networks (CNNs) for automatic classification and diagnosis of medical images have recently displayed a remarkable performance. However, the CNNs fail to recognize original images rotated and oriented in different directions which limits its performance. This paper presents a new capsule network (CapsNet) based framework known as multi-lane atrous feature fusion capsule network (MLAF-CapsNet) for brain tumor type classification. The MLAF-CapsNet consistsof atrous and CLAHE, where the atrous increases receptive fields and maintains spatial representation, whereas the CLAHE is used as a base layer which uses an improved adaptive histogram equalization (AHE) to enhance the color of the input images.The proposed method is evaluated using whole brain tumor and segmented tumor datasets. The efficiency performance of thetwo datasets is explored and compared. The experimental results of the MLAF-CapsNet show better accuracies (93.40% and 96.60%) and precisions (94.21% and 96.55%) in feature extraction based on the original images from the two datasets than the traditional CapsNet (78.93% and 97.30%). Based on the augmentation from the two datasets, the proposed method achieved the best accuracies (98.48% and 98.82%) and precisions (98.88% and 98.58%) in extracting features compared to the traditional CapsNet. Our results indicate that the proposed method can successfully improve brain tumor classification problem and can support radiologist in medical diagnostics.

## Baseline Capsule Network
### by Sabour et al. 2017

![capp](https://user-images.githubusercontent.com/33870014/111817413-f2c2b880-8918-11eb-840b-983376725e45.jpg)


## MLAF-CapsNet
![mlaf](https://user-images.githubusercontent.com/33870014/111817353-e2124280-8918-11eb-88e4-ffaa056cf6f3.png)


#### Train the model

python train.py or run in jupyter notebook with th command %run train.py

#### Original Code
The code used for this study is a modification of the code at https://github.com/XifengGuo/CapsNet-Keras.
