import os
import torch
import argparse
import numpy as np

import torchvision.datasets as datasets
from training import validate, accuracy
import torchvision.transforms as transforms
from models import MobileNetV1_Stud, MobileNetV1_Teach
from torch import nn

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

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
args = parser.parse_args()

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

    #validate
    acc1, acc5 = validate(val_loader, model, criterion, num_classes, args)
    dic_temps[file] = (format(acc1.item(), '.2f'), format(acc5.item(), '.2f'))
    print(acc1, acc5)

#np.save('./results/all_accuracy.npy', dic_temps)
accuracy = np.load('./results/all_accuracy.npy')

print(accuracy)

    