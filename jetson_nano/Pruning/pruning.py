from ast import operator
from asyncio import subprocess
from copyreg import remove_extension
from this import d
from turtle import update
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
import numpy as np
import torch.onnx
import subprocess as sp

sys.path.append('../jetson_benchmarks/benchmarks_pt2/')
from obj_detection_ssd_custom_utils import get_fps

sys.path.append('/home/flavio/thesis/jetson_nano/jetson-inference/build/aarch64/bin/')
#import detectnet

from prettytable import PrettyTable
from tqdm import tqdm
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from loaders import *
from vision.ssd.config import mobilenetv1_ssd_config
from vision.nn.multibox_loss import MultiboxLoss
from train_ssd.train_ssd import test
from torch import nn
#from ..jetson_benchmarks.benchmarks_pt2.obj_detection_ssd_custom_utils import get_fps

#from torch import fx
#from torch.autograd import Variable
#from soft import simplify 
#from soft.simplify.utils import get_bn_folding, get_pinned_out, get_pinned
#from soft.simplify import remove_zeroed
#from soft.simplify.remove import remove_zeroed
#from soft.simplify import simplify as sp
#from torchsummary import summary
#from train_ssd.vision.nn.mobilenet import MobileNetV1
#from soft.simplify import utils as utl

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

methods = ["unstructured", "structured", 'global']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


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

def get_previous_layer(node, modules):
    # print("get_previous_layer")
    for input_node in node.all_input_nodes:
        # print(input_node.name)
        if input_node.target in modules and isinstance(modules[input_node.target], (nn.Conv2d, nn.BatchNorm2d)):
            return input_node.target
        else:
            return get_previous_layer(input_node, modules)

def prune_net_local(val_pruned, method):
    net = load_net()
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            #print("Name module: ", name)

            if method == 'unstructured':
                prune.l1_unstructured(module, name='weight', amount=val_pruned/100)
                #prune.l1_unstructured(module, name='bias', amount=3)

            elif method == 'structured':
                prune.ln_structured(module, 'weight', val_pruned/100, n=1, dim=1)
            
            prune.remove(module, 'weight') #remove mask
            print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement())))

    return net


def generate_pruned_models(start, total_percentage, step, model_dir):
    print("Generating pruned models...")
    with tqdm(total=total_percentage/step, file=sys.stdout) as pbar:
        logging.disable(logging.CRITICAL)
        for i in range(start, step+total_percentage, step):
            sys.stdout = open(os.devnull, 'w')
            pruned_amount = i
            net_local_unstructured = prune_net_local(pruned_amount, 'unstructured')
            net_local_unstructured.save('./models/'+model_dir+'/unstructured/' + str(pruned_amount) + '%_unstructured_pruned_model.pth')

            net_local_structured = prune_net_local(pruned_amount, 'structured')
            net_local_structured.save('./models/'+model_dir+'/structured/' + str(pruned_amount) + '%_structured_pruned_model.pth')

            net_global= prune_net_global(pruned_amount)
            net_global.save('./models/'+model_dir+'/global/' + str(pruned_amount) + '%_global_pruned_model.pth')
            
            sys.stdout = sys.__stdout__
            pbar.update(1)
            
#get all models pruned
# start = 0 #initial amout of pruning
# step = 5 
# stop = 100 #final amout of pruning (step excluded)
# model_dir = 'pruned'
# generate_pruned_models(start, stop, step, model_dir)

#evaluation loss of any pruned models
def get_loss():
    dict_loss = {k: [] for k in methods}

    path_prn = './models/pruned/'
    config = mobilenetv1_ssd_config
    _, val_loader, _ = get_loaders(4,4)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                center_variance=0.1, size_variance=0.2, device=DEVICE)

    print("Computing losses values...")
    with tqdm(total=len(os.listdir(path_prn+methods[0]))*len(methods), file=sys.stdout) as pbar:    
        for method in methods:
            path_single_method = path_prn+method
            if os.path.isdir(path_single_method):
                    for i in range(0, 105, 5):
                        path_file = str(i)+'%_'+method+'_pruned_model.pth'
                        logging.disable(logging.CRITICAL)
                        sys.stdout = open(os.devnull, 'w')
                        net = create_model()
                        net.load(path_single_method+'/'+path_file)
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

#create dict_loss
#get_loss()

def get_plot(method, index):
    file = open("dict_loss.pkl", "rb")
    dict_loss = pickle.load(file)
    plt.style.use('seaborn-white')

    losses = []
    for i in range (0, len(dict_loss[method])):
        total_amount = i * 5
        vl = dict_loss[method]
        loss = vl[i][index] #get the loss of currente percentage
        losses.append((total_amount/100,loss))

    pd.DataFrame.assign

    (pd.DataFrame(losses, columns=['sparsity', 'loss'])
    .pipe(lambda df: df.assign(perf=(df.loss - pd.Series([losses[0][1]] * len(df))) / losses[0][1] + 1))
    .plot.line(x='sparsity', y='perf', figsize=(12, 8), title="L1 " + method +  " Pruned Mean Batch Loss"))
    sns.despine()

    if index==0:
        plt.savefig('images/'+method+"_Loss_"+"_.png")
    elif index==1:
        plt.savefig('images/'+method+"_ClassLoss_"+"_.png")
    elif index==2:
        plt.savefig('images/'+method+"_RegrLoss_"+"_.png")

#generating loss plot: Global loss, Regression loss and classification loss.
# for i in range(0,3):
#     get_plot(methods[i], 0)
#     get_plot(methods[i], 1)
#     get_plot(methods[i], 2)

#Converting pruned model in onnx format

def convert_models():
    path_models = './models/pruned/'

    with tqdm(total=(len(os.listdir(path_models+methods[0]))-1)*len(methods), file=sys.stdout) as pbar:
        for i in range(0,3):
            directory = methods[i]

        dir_input = path_models+directory
        dir_output = dir_input + '/' + "onnx"

        if not os.path.exists(dir_output):
            os.makedirs(dir_output)
        else:
            os.removedirs(dir_output)
            os.makedirs(dir_output)

        print("Converting models...")
        for i in range(0, 105, 5):
            logging.disable(logging.CRITICAL)
            sys.stdout = open(os.devnull, 'w')
            model_path_pth = dir_input+"/"+str(i)+'%_'+directory+'_pruned_model.pth'
            model_path_onnx = dir_output+"/"+str(i)+'%_'+directory+'_pruned_model.onnx'
            os.system(
            "cd train_ssd; python3 onnx_export.py --input="+ model_path_pth + " " + 
            "--output=" + model_path_onnx + " " + 
            "--model-dir=" + path_models + " >/dev/null 2>&1")
            sys.stdout = sys.__stdout__
            pbar.update(1)

#convert_models()

#test performance

def get_models(method, dir):
    files = []
    dir_input = './models/pruned/'+method+'/'+dir
    for i in range(0, 105, 5):
        model_path_pth = dir_input+"/"+str(i)+'%_'+method+'_pruned_model.'+dir
        files.append(model_path_pth)
    return files
    
for i in range(2,3):
    input_list = [
        #"video/240p_60fps.mp4",
        #"video/360p_30fps.mp4"
        #"video/480p_30fps.mp4",
        #"video/720p_30fps.mp4",
        # "video/1080p_30fps.mp4",
        # "video/1080p_60fps.mp4",
        "csi://0"
        #"/dev/video1"
    ]
    output = "display://0"
    networks_list = get_models(methods[i], 'onnx')
    #half_list_net = [networks_list[i] for i in range(0, int(len(networks_list)/2))]
    operation = './benchmarks/'+methods[i]
    labels = './models/pruned/labels.txt'
    print("Get FPS for " + methods[i] + " method...")

    #cmd = 'cd /home/flavio/thesis/jetson_nano/jetson-inference/build/aarch64/bin/;'
    #cmd2 = 'python3 detectnet.py --model=/home/flavio/thesis/jetson_nano/Pruning/models/pruned/unstructured/onnx/50%_unstructured_pruned_model.onnx --labels=/home/flavio/thesis/jetson_nano/Pruning/models/pruned/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /home/flavio/thesis/jetson_nano/Pruning/data/OpenImages/test /home/flavio/thesis/jetson_nano/Pruning/data/OpenImages/result'
    
    #sts = os.system(cmd+cmd2)
    # sts2 = sp.Popen(cmd+cmd2, shell=True, stdout=subprocess.PIPE).stdout
    # info = sts2.read()
    # print(type(info))
    # print(type(info.decode()))
    # print(info.decode())
    dataframe, filename = get_fps(input_list, output, networks_list, labels)
    path_dataset = operation+'/'+filename
    dataframe.to_csv(operation+'/'+filename)
    # if not os.path.exists(path_dataset):
    #         with open(path_dataset, "rb") as f:
    #             dataframe_old = pickle.load(f)

    # else:
    #     dataframe.to_csv(operation+'/'+filename)

    
