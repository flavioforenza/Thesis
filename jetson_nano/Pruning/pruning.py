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

def load_net():
    path_virgin_model = './models/original_SSD_Mobilenet_V1.pth'
    path_checkpoints = './models/checkpoints/'
    net = create_model()
    if os.path.isfile(path_virgin_model):
        #if exists a pretrained model
        if os.path.isdir('./models/checkpoints/'):
            for file in os.listdir('./models/checkpoints/'):
                if file.endswith(".pth"):
                    net.load(path_checkpoints+file)
                else:
                    print("No checkpoint models exist!")
                    return 0
        else: #get the virgin model
            net.load(path_virgin_model)
    return net

def prune_net_global(val_pruned):
    net = load_net()
    module_tups = []
    for _, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_tups.append((module,'weight'))
    
    prune.global_unstructured(
        parameters=module_tups, 
        pruning_method=prune.L1Unstructured,
        amount=val_pruned/100
    )

    print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement())))

    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return net

def prune_net_local(val_pruned, method):
    net = load_net()
    for _, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if method == 'unstructured':
                prune.l1_unstructured(module, name='weight', amount=val_pruned/100)
                #prune.l1_unstructured(module, name='bias', amount=3)

            elif method == 'structured':
                prune.ln_structured(module, 'weight', val_pruned/100, n=1, dim=1)
            
            print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement())))
            prune.remove(module, 'weight') #remove mask
    return net

for i in range(10, 80):
    pruned_amount = i
    #net_local_unstructured = prune_net_local(pruned_amount, 'unstructured')
    #net_local_unstructured.save('./models/pruned/local-unstructured/' + str(pruned_amount) + '%_unstructured_pruned_model.pth')

    #net_local_structured = prune_net_local(pruned_amount, 'structured')
    #net_local_structured.save('./models/pruned/local-structured/' + str(pruned_amount) + '%_structured_pruned_model.pth')

    net_global= prune_net_global(pruned_amount)
    net_global.save('./models/pruned/global/' + str(pruned_amount) + '%_global_pruned_model.pth')