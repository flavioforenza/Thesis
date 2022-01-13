import torch
import torch.nn.utils.prune as prune
import sys
import logging
import os.path
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from loaders import *
from vision.ssd.config import mobilenetv1_ssd_config


#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Using CUDA...")

#getting the new model (SSD-Mobilenet-V1)
def create_model():
    _, _, num_classes = get_loaders(4,4)
    model = create_mobilenetv1_ssd
    net = model(num_classes)
    #net.save(path)
    return net

def prune_net(val_pruned):
    path_virgin_model = './models/original_SSD_Mobilenet_V1.pth'
    path_checkpoints = './models/checkpoints/'
    net = create_model()
    if os.path.isfile(path_virgin_model):
        #if exists a pretrained model
        if os.path.isdir('./models/checkpoints/'):
            for file in os.listdir('./models/checkpoints/'):
                if file.endswith(".pth"):
                    net.load(path_checkpoints+file)
        else: #get the virgin model
            net.load(path_virgin_model)

    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            #print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement())))
            #print("####### BEFORE #######")
            print("Name:", name) 
            print("Module", module)
            #print(list(module.named_parameters()))
            #print(module.weight)
            #prune.l1_unstructured(module, name='weight', amount=val_pruned/100)

            prune.ln_structured(module, 'weight', val_pruned/100, n=2, dim=1)

            #prune.l1_unstructured(module, name='bias', amount=3)
            #print("####### AFTER #######")
            #print(module.weight)
            prune.remove(module, 'weight') #remove mask
            #prune.remove(module, 'bias') #remove mask
            print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement())))

    return net

for i in range(10, 80):
    pruned_amount = i
    net = prune_net(pruned_amount)
    net.save('./models/pruned' + str(pruned_amount) + '%_pruned_model.pth')