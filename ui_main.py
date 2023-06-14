import streamlit as st
import cv2
import json
import math, random
import matplotlib.pyplot as plt 
import numpy as np
import os
# from shapely.geometry import Polygon
import sys
import time
from typing import List, Tuple
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import transforms, utils
from torchvision.datasets import CocoDetection
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as transforms

from PIL import Image
import argparse

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set

import logging


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
 
def get_prediction(img, threshold, model):
#     transform = transforms.Compose([transforms.ToTensor()])
#     img = transform(img)
    pred = model([img])

    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]

    masks = masks[:pred_t+1]

    pred_class = pred_class[:pred_t+1]
    return masks, pred_class

def white_out(img_tensor, model, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    img = img_tensor.numpy()
    img = np.transpose(img, (1, 2, 0))
    img_new = img.copy()
    classes = set()
    print("available classes to delete. Choose one")
    for i in range(len(masks)):
        classes.add(pred_cls[i])

    [print(i) for i in list(classes)]
  
    class_name = list(classes)[0]# input()
    for i in range(len(masks)):
        if pred_cls[i] == class_name:
            img_new[masks[i] != 0] = 1
  
    return img_new


class Args:
    cuda = True
    seed = 123
    
opt = Args()
device = torch.device("cuda:0" if opt.cuda else "cpu")

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def transform_image(image):
    transform_list = [
        transforms.ToTensor(),
#         transforms.Resize(size=(256, 256)),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    transformed_img = transform(image).to(device)
    return transformed_img


def main():
    st.title("Instance Eraser Demo")
    st.write("Upload an image")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        transform_white = transforms.Compose([transforms.ToTensor()])
        transformed_image = transform_white(image)
        masks, pred_cls = get_prediction(transformed_image, 0.5, model)
        classes = set()
        for i in range(len(masks)):
            classes.add(pred_cls[i])
        options = list(classes)
        selected_option = st.selectbox("Available classes to delete. Choose one", options)
        st.write("You selected:", selected_option)
        transformed_img = transformed_image.numpy()
        transformed_img = np.transpose(transformed_img, (1, 2, 0))
        transformed_img_new = transformed_img.copy()
        for i, mask in enumerate(range(len(masks))):
            if pred_cls[i] == selected_option:
                transformed_img_new[masks[i] != 0] = 1
                break
        st.image(transformed_img_new, caption="Uploaded Image", use_column_width=True, clamp=True)
        generator_input = transform_image(transformed_img_new)
        generator_input = generator_input.view(1, generator_input.shape[0], generator_input.shape[1], generator_input.shape[2])
        prediction = net_g(generator_input)
        prediction = prediction[0]
        transform = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),  # Rescale the values to 0-1 range
            transforms.ToPILImage()
        ])

        image_pil = transform(prediction)
#         st.image(image_pil)
        st.image([image, transformed_img_new, image_pil], width=200)
        
        
        

# Run the app
if __name__ == "__main__":
    coco_weight_path = "checkpoints_new_256/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load(coco_weight_path))
    model.eval()
    net_g = torch.load("checkpoints_new_256/netG_model_epoch_30.pth",map_location=device)
    # net_g.eval()
    main()
    