import os
import torch
import argparse
import numpy as np
import logging
import sys
import pickle
import matplotlib.pyplot as plt


import torchvision.datasets as datasets
from training import validate, accuracy
import torchvision.transforms as transforms
from models import MobileNetV1_Stud, MobileNetV1_Teach
from torch import nn
from prettytable import PrettyTable
from train_sd.train_ssd import test
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from torch.utils.data import DataLoader, ConcatDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.ssd import MatchPrior
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.nn.multibox_loss import MultiboxLoss
from mobilenet_ssd import create_mobilenetv1_ssd as ssd_student



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--datasets', '--data', nargs='+', default=["/home/flavio/thesis/jetson_nano/train-ssd/data/OpenImages"], help='Dataset directory path')
parser.add_argument('--balance-data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--checkpoint-folder', '--model-dir', default='checkpoints_ssd_distill/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--batch-size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--num-workers', '--workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--use-cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

args = parser.parse_args()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def get_accuracy_BaseNet():
    valdir = os.path.join('./data/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=8, shuffle=False,
            num_workers=4, pin_memory=True)


    temps = [1,2,3,4,5,10,15]
    model_paths = ['./model/teacher-1000.pth', './students_distilled/student-1000.pth']
    CHECKPOINTS_DIR = './students_distilled'
    prefix = 'student_distill-1000'
    criterion = nn.CrossEntropyLoss().cuda(0)
    num_classes = 8

    for epoch_count in temps:
        model_paths.append(os.path.join(CHECKPOINTS_DIR, '{}_T={}.pth'.format(prefix, epoch_count)))

    dic_temps = {}

    for file in model_paths:
        print(file)
        if 'teacher' in file:
            model = MobileNetV1_Teach(num_classes)
        else:
            model = MobileNetV1_Stud(num_classes, 0.25)
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint.state_dict())

        model.cuda(args.gpu)

        #param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters: ", count_parameters(model))

        #validate
        acc1, acc5 = validate(val_loader, model, criterion, num_classes, args)
        dic_temps[file] = (format(acc1.item(), '.2f'), format(acc5.item(), '.2f'))
    
    #np.save('./results/all_accuracy.npy', dic_temps)
    #accuracy = np.load('./results/all_accuracy.npy')

    acc_file = open("./results/all_accuracy_temps.pkl", "wb")
    pickle.dump(dic_temps, acc_file)
    acc_file.close()
    
    return dic_temps

#acc_temps = get_accuracy_BaseNet()
# file = open("./results/all_accuracy_temps.pkl", "rb")
# acc_temps = pickle.load(file)
#print(acc_temps)

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def SSD_loss(path, loader, criterion, device, num_classes):
    if 'teacher' in path:
        create_net = create_mobilenetv1_ssd
        model = create_net(num_classes)
    else:
        create_net = ssd_student
        alpha = 0.25
        model = create_net(num_classes, alpha)
    model.load(path)
    model.cuda()

    #SSD parameters
    print(count_parameters(model))

    val_loss, val_regression_loss, val_classification_loss = test(loader, model, criterion, device)
    logging.info(
        f"Validation Loss: {val_loss:.4f}, " +
        f"Validation Regression Loss {val_regression_loss:.4f}, " +
        f"Validation Classification Loss: {val_classification_loss:.4f}"
    )

#load SSD model
path_teacher = './model/teacher-ssd-1000.pth'
path_student_Freeze = './model/mb1-ssd-Epoch-500-Loss-4.808817083185369.pth'
path_student_noFreeze = './model/mb1-ssd-Epoch-500-Loss-3.7902622743086383.pth'

config = mobilenetv1_ssd_config

train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

for dataset_path in args.datasets:
    dataset = OpenImagesDataset(dataset_path,
            transform=train_transform, target_transform=target_transform,
            dataset_type="train", balance_data=args.balance_data)

    label_file = os.path.join(args.checkpoint_folder, "labels.txt")
    store_labels(label_file, dataset.class_names)
    logging.info(dataset)
    num_classes = len(dataset.class_names)

val_dataset = OpenImagesDataset(dataset_path,
                                transform=test_transform, target_transform=target_transform,
                                dataset_type="test")

val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)

SSD_loss(path_student_noFreeze, val_loader, criterion, DEVICE, num_classes)
