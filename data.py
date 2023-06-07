from os.path import join

from dataset import BackgroundDataset


def get_training_set(root_dir = "/datasets/COCO-2017/train2017/",
                 label_file = "/datasets/COCO-2017/anno2017/instances_train2017.json",
                 transform = None
                 ):

    return BackgroundDataset(root_dir, label_file, transform)


def get_test_set(root_dir = "/datasets/COCO-2017/val2017/",
                 label_file = "/datasets/COCO-2017/anno2017/instances_val2017.json",
                 transform = None
                 ):

    return BackgroundDataset(root_dir, label_file, transform)
