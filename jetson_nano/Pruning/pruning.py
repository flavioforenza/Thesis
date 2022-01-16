import torch
import torch.nn.utils.prune as prune
import sys
import logging
import os.path
import os 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tqdm import tqdm 
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from loaders import *
from vision.ssd.config import mobilenetv1_ssd_config
from vision.nn.multibox_loss import MultiboxLoss
from train_ssd.train_ssd import test

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

#if torch.cuda.is_available():
#    torch.backends.cudnn.benchmark = True
#    logging.info("Using CUDA...")

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


def generate_pruned_models(start, total_percentage, step):
    print("Generating the pruned models...")
    with tqdm(total=(total_percentage-step)/step, file=sys.stdout) as pbar:
        logging.disable(logging.CRITICAL)
        for i in range(start, step+total_percentage, step):
            sys.stdout = open(os.devnull, 'w')
            pruned_amount = i
            net_local_unstructured = prune_net_local(pruned_amount, 'unstructured')
            net_local_unstructured.save('./models/pruned/unstructured/' + str(pruned_amount) + '%_unstructured_pruned_model.pth')

            net_local_structured = prune_net_local(pruned_amount, 'structured')
            net_local_structured.save('./models/pruned/structured/' + str(pruned_amount) + '%_structured_pruned_model.pth')

            net_global= prune_net_global(pruned_amount)
            net_global.save('./models/pruned/global/' + str(pruned_amount) + '%_global_pruned_model.pth')
            
            sys.stdout = sys.__stdout__
            pbar.update(1)
            
#get all models pruned
#start = 0 #initial amout of pruning
#step = 5 
#stop = 100 + step #final amout of pruning (step excluded)
#generate_pruned_models(start, stop, step)

#evaluation loss of any pruned models
def get_loss():
    methods = ["unstructured", "structured", 'global']
    dict_loss = {k: [] for k in methods}

    path_prn = './models/pruned/'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = mobilenetv1_ssd_config
    _, val_loader, _ = get_loaders(4,4)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                center_variance=0.1, size_variance=0.2, device=DEVICE)

    with tqdm(total=len(os.listdir(path_prn+methods[0]))*len(methods), file=sys.stdout) as pbar:
        for method in methods:
            path_single_method = path_prn+method
            if os.path.isdir(path_single_method):
                    for file in os.listdir(path_single_method):
                        logging.disable(logging.CRITICAL)
                        sys.stdout = open(os.devnull, 'w')
                        net = create_model()
                        net.load(path_single_method+'/'+file)
                        net.cuda()
                        loss, regression_loss, classification_loss = test(val_loader, net, criterion, DEVICE)
                        #print("Validation Loss:", loss)
                        #print("Validation Regression Loss", regression_loss)
                        #print("Validation Classification Loss:", classification_loss)
                        dict_loss[method].append([loss, regression_loss, classification_loss])
                        sys.stdout = sys.__stdout__
                        pbar.update(1)
    a_file = open("dict_loss.pkl", "wb")
    pickle.dump(dict_loss, a_file)
    a_file.close()


def get_plot(method, index):
    file = open("dict_loss.pkl", "rb")
    dict_loss = pickle.load(file)
    plt.style.use('seaborn-white')

    losses = []
    for i in range (0, len(dict_loss[method])):
        total_amount = i * 5
        vl = dict_loss[method]
        loss = vl[i][index] #get the loss of currente percentage
        losses.append((total_amount,loss))

    pd.DataFrame.assign

    (pd.DataFrame(losses, columns=['sparsity', 'loss']).pipe(lambda df: df.assign(perf=(df.loss - pd.Series([losses[0][1]] * len(df))) / losses[0][1] + 1)).head(15).plot.line(x='sparsity', y='perf', figsize=(12, 8), title="L1 Unstructured Pruned Mean Batch Loss"))
    sns.despine()

    plt.savefig("out.png")

get_plot('unstructured', 0)