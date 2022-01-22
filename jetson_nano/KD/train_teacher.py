from turtle import back
import torch
import os
import sys
 
from models import MobileNetV1_Teach
from train_sd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

def get_TeachModel():
    #pre-trained model (1000 epochs)
    path_preTrain_model = '/home/flavio/thesis/jetson_nano/Pruning/models/checkpoints/mb1-1000.pth'
    if os.path.exists(path_preTrain_model):
        model = create_mobilenetv1_ssd
        net = model(9)
        net.load(path_preTrain_model)
        return net
    else:
        path_model = './model/teacher.pth'
        teacher_model = MobileNetV1_Teach(9).model
        torch.save(teacher_model.state_dict(), path_model)
        return teacher_model

#load SSD Model
net = get_TeachModel()
#teacher net as the backbone net of the SSD net
teacher = net.base_net

#save the backbone model
torch.save(teacher.state_dict(), './model/teacher.pth')

#provo ad allenare la rete con il train.py datomi da dusty



