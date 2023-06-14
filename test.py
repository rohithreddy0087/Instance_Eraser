from __future__ import print_function
import os

import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

class Args:
    cuda = True

if __name__ == "__main__":
    opt = Args()
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    image_dir = "test_images/"
    model_path = "checkpoints/netG_model_epoch_1.pth"
    net_g = torch.load(model_path).to(device)
    image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
    transform_list = [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    for image_name in image_filenames:
        img = load_img(image_dir + image_name)
        img = transform(img)
        input = img.unsqueeze(0).to(device)
        out = net_g(input)
        out_img = out.detach().squeeze(0).cpu()
        save_img(out_img, "result/{}".format(image_name))
