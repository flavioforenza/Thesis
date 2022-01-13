import torch
import torchvision
import os

from vision.datasets.open_images import OpenImagesDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd import MatchPrior
from vision.ssd.config import mobilenetv1_ssd_config


from torch.utils.data import DataLoader, ConcatDataset

def get_loaders(batch_size_train, batch_size_test):

    config = mobilenetv1_ssd_config

    #DEFINE TRAINING DATALOADER 
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    path_dataset = ['/home/flavio/thesis/jetson_nano/train-ssd/data/OpenImages/']

    datasets = []
    for dataset_path in path_dataset:
        dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=True)
        datasets.append(dataset)

    train_dataset = ConcatDataset(datasets)

    train_loader = DataLoader(train_dataset, batch_size_train,
                              num_workers=4,
                              shuffle=True)

    #DEFINE VALIDATION DATALOADER                          

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
                            
    val_dataset = OpenImagesDataset(dataset_path,
                                    transform=test_transform, target_transform=target_transform,
                                    dataset_type="test")                             

    val_loader = DataLoader(val_dataset, batch_size_test,
                            num_workers=4,
                            shuffle=False)

    return train_loader, val_loader, len(dataset.class_names) #num_classes

