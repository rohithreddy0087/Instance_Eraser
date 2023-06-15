# Instance Eraser
Our project aims to develop an algorithm inspired by Google's MagicEraser feature to remove specified objects from images and seamlessly replace them with background elements. We provide a user-friendly interface where users can upload their image and select the object class to remove. The algorithm involves instance segmentation, removing instances of the specified object category, and reconstructing the image by filling the removed areas with appropriate background information. This algorithm addresses the need to remove unwanted objects while preserving the background, benefiting applications such as landscape photography, medical imaging, and satellite imaging.

## Method
![Flowchart](https://github.com/rohithreddy0087/Instance_Eraser/assets/51110057/897e56da-c10c-48d0-987a-aebb7780467a)
Our method for developing the "Instance Eraser" algorithm involves the following steps:

1. Dataset Generation: We utilize existing benchmark datasets for training the image segmentation model. Additionally, we modify the dataset by randomly whitening out background regions to create inputs for training the image inpainting models.
![dataset_flowchart](https://github.com/rohithreddy0087/Instance_Eraser/assets/51110057/5efafffd-e541-40af-853f-a0199e761f7d)

3. Instance Segmentation: We employ techniques like Mask R-CNN with a U-Net decoder or pretrained Mask R-CNN models to segment the input image into different regions. For UI we are using pretrained MaskRCNN as it performed better compared to out trained MaskRCNN with U-Net model.

4. Instance Removal: Users specify the object class they want to remove, and based on the image segmentation results, we identify and remove all instances of the specified object category from the image. The removed object pixels are replaced with white background pixels.

5. Image Reconstruction: The instance-removed image serves as input to generative models like Pix2Pix (supervised learning) with a ResNet generator, Pix2Pix with a U-Net generator, or CycleGAN (unsupervised learning). These models are trained to reconstruct the image by filling the removed areas with meaningful background information while preserving visual coherence.

We experimented with all the mentioned models and the best model which was able to recover background was Pix2pix with ResNet generator. Although we were not able to achieve state of the art results, we were able to recover the background which is contextually relevant. With more experimentation and training we can achieve good results.

## Requirements
