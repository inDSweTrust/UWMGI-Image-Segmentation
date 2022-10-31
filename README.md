# UWMGI-Image-Segmentation
## Summary

This repository contains the full training-inference image segmentation pipeline for the [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview) Kaggle competition. 

The goal was to create a baseline image segmentation pipeline to use in future contests or other applications.

This is a Unet based model with EfficientNet backbone pre-trained on ImageNet evaluated with Dice Coefficient and 3D Hausdorff Distance. Model segments stomach and intestines on MRI scans.

![image](https://user-images.githubusercontent.com/68122114/194739608-be8ddb9f-55c4-407b-baf7-9a2433551929.png)
![image](https://user-images.githubusercontent.com/68122114/194739630-ce028aa1-b781-4bd4-9505-24b4e0f5e8ce.png)

## Sources
* https://github.com/milesial/Pytorch-UNet
* https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet
* https://www.image-net.org/index.php

