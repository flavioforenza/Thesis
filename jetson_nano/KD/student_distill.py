import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import scheduler
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

from mobilenet import Net
from training import AverageMeter, ProgressMeter, accuracy, save_checkpoint
from models import MobileNetV1_Stud, MobileNetV1_Teach

class CrossEntropyLossForSoftTarget(nn.Module):
    def __init__(self, T=20):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.T*self.T)

def train(train_loader, student_model, criterion, optimizer, epoch, num_classes, teacher_model):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student_model.train()

    # get the start time
    epoch_start = time.time()
    end = epoch_start

    total_loss = 0.0
    # train over each image batch from the dataset
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        gpu = 0
        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output_student = student_model(images)
        output_teacher = teacher_model(images)
        
        loss_teacher = criterion_soft(output_student, output_teacher)
        loss = criterion(output_student, target) + loss_teacher 

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_student, target, topk=(1, min(5, num_classes)))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clip weight
        # max_norm = 15.0
        # named_parameters = dict(student_model.named_parameters())
        # for layer_name in ['layer1', 'layer2', 'layer3']:
        #     with torch.no_grad():
        #         weight = named_parameters['{}.weight'.format(layer_name)]
        #         bias = named_parameters['{}.bias'.format(layer_name)].unsqueeze(1)
        #         weight_bias = torch.cat((weight, bias),dim=1)
        #         norm = torch.norm(weight_bias, dim=1, keepdim=True).add_(1e-6)
        #         clip_coef = norm.reciprocal_().mul_(max_norm).clamp_(max=1.0)
        #         weight.mul_(clip_coef)
        #         bias.mul_(clip_coef)

        total_loss += loss.item()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
    
    print("Epoch: [{:d}] completed, elapsed time {:6.3f} seconds".format(epoch, time.time() - epoch_start))

    


#DATASET_DIR = '../mnist/'

device = torch.device('cuda')
print(device)

# Load the best teacher model
teacher_model = torch.load('./model/teacher-180.pth')
#teacher_model = Net(8)
#teacher_model.load_state_dict(torch.load('./model/teacher-180.pth'))
teacher_model.eval()

student_model = MobileNetV1_Stud(8)

criterion = nn.CrossEntropyLoss(reduction='mean')
criterion_soft = CrossEntropyLossForSoftTarget()

optimizer = optim.SGD(student_model.parameters(), lr=0.1)

lr_init = 0.1
gamma = 0.998
lrs = np.zeros(shape=(1000,))
lr = lr_init

for step in range(1000):
    lrs[step] = lr
    lr *= gamma
momentums = np.concatenate([np.linspace(0.5, 0.99, 500), np.full(shape=(2500,), fill_value=0.99)])
list_lr_momentum_scheduler = scheduler.ListScheduler(optimizer, lrs=lrs, momentums=momentums)


traindir = os.path.join('./data/', 'train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        #transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

num_classes = len(train_dataset.classes)
gpu=0

train_sampler = None

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, sampler=train_sampler)

cudnn.benchmark = True

student_model.train()

if gpu is not None:
    torch.cuda.set_device(gpu)
    student_model = student_model.cuda(gpu)
    teacher_model = student_model.cuda(gpu)

for epoch in range(0, 1000):
    train(train_loader, student_model, criterion, optimizer, epoch, num_classes, teacher_model)

    
    # save_checkpoint({
    #     'model': model,
    #     'model_name': model_name,
    #     'epoch': epoch + 1,
    #     'arch': args.arch,
    #     'resolution': args.resolution,
    #     'num_classes': num_classes,
    #     'state_dict': model.state_dict(),
    #     'optimizer' : optimizer.state_dict(),
    # }, args)

    # remember best acc@1 and save checkpoint
    #is_best = acc1 > best_acc1
    #best_acc1 = max(acc1, best_acc1)

