from ast import operator
from asyncio import subprocess
from copyreg import remove_extension
from pyexpat import model
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
import glob
import gzip

#sys.path.append('../jetson_benchmarks/benchmarks_pt2/')
#from obj_detection_ssd_custom_utils import get_fps

#sys.path.append('/home/flavio/thesis/jetson_nano/jetson-inference/build/aarch64/bin/')
#import detectnet

#sys.path.append('/home/flavio/thesis/jetson_nano/KD/')
#import statistics as st


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
#from train_ssd.vision.nn.mobilenet import MobileNetV1

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

methods = ["unstructured", "structured", 'global']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def difference(start, end):
    return ((end-start)/start)*100

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

def get_plot(method, index, t_loss):
    file = open("dict_loss.pkl", "rb")
    dict_loss = pickle.load(file)
    plt.style.use('seaborn-white')

    losses = []
    for i in range (0, len(dict_loss[method])):
        total_amount = i * 5
        vl = dict_loss[method]
        loss = vl[i][index] #get the loss of current percentage
        losses.append((total_amount/100,loss))

    ref_value = losses[0][1]
    lst_diff_loss = [(0,0)]
    for i in range (1, len(losses)):
        diff = difference(ref_value, losses[i][1])
        lst_diff_loss.append((losses[i][0]*100, diff))

    #disp_list = y_lab[0::2]

    plt.plot([fisrt[0] for fisrt in losses], [second[1] for second in lst_diff_loss])
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Losses')

    a_file = open("lst_perc_loss_"+method+"_"+t_loss+".pkl", "wb")
    pickle.dump(lst_diff_loss, a_file)
    a_file.close()
    
    # pd.DataFrame.assign

    # (pd.DataFrame(losses, columns=['Sparsity', 'loss'])
    # .pipe(lambda df: df.assign(Loss=(df.loss - pd.Series([losses[0][1]] * len(df))) / losses[0][1] + 1))
    # .plot.line(x='Sparsity', y='Loss', figsize=(9, 6),linewidth=2.5, title="L1 " + method +  " " + t_loss))
    # plt.legend(loc='upper left')
    # sns.despine()

    # if index==0:
    #     plt.savefig('images/'+method+"_Loss.png")
    # elif index==1:
    #     plt.savefig('images/'+method+"_RegrLoss.png")
    # elif index==2:
    #     plt.savefig('images/'+method+"_ClassLoss.png")

#generating loss plot: Global loss, Regression loss and classification loss.
# for i in range(1,3):
#     get_plot(methods[i], 0, 'Loss')
#     get_plot(methods[i], 1, 'Regression')
#     get_plot(methods[i], 2, 'Classification')

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

def zip_model():
    path = './models/pruned'
    dict_size = {k:[] for k in methods}
    for filename in os.listdir(path):
        method = os.path.join(path, filename)
        for i in range(0, 105, 5):
            model_path_pth = method+"/"+str(i)+'%_'+filename+'_pruned_model.pth'
            #file_stats = os.path.getsize(model_path_pth)
            #original_size = float(format(file_stats*10e-7, '.1f'))
            #print("Original Size: ", original_size)
            in_data = open(model_path_pth, "rb").read()
            gzf = gzip.open(model_path_pth+'.gz', "wb")
            gzf.write(in_data)
            
            file_stats_new = os.path.getsize(model_path_pth+'.gz')
            pruned_size = float(format(file_stats_new*10e-7, '.1f'))
            dict_size[filename].append(pruned_size)
            print("Model: ", model_path_pth)
            print("Pruned Size: ", pruned_size)

            gzf.close()
    return dict_size
    
# dict_size_zip = zip_model()
# a_file = open("dict_zip.pkl", "wb")
# pickle.dump(dict_size_zip, a_file)
# a_file.close()

def get_plot_size(value, x):
    width = 7
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()

    ax.bar(x, value, width, color="blue", align='center')

    rects = ax.patches
    
    idx = 0
    for rect, label in zip(rects, value):
       height = rect.get_height()
       ax.text(rect.get_x() + rect.get_width() / 2, height, str(int(difference(30.7, label))) + "%", ha="center", va="bottom", color='green', fontweight='bold', fontsize=14)
       ax.text(rect.get_x() + rect.get_width() / 2, height -1.5, str(label), ha="center", va="bottom", color='white', fontweight='bold', fontsize=11)
       idx += 1

    plt.xlabel('% Sparsity', fontweight='bold')
    plt.ylabel('MByte', fontweight='bold')
    plt.title('Compressed pruned model size.')
    plt.yticks(np.arange(0, max(value)+5, 5))
    #plt.show()
    plt.savefig('./images/Unstructured_reduction.png')

#file = open("dict_zip.pkl", "rb")
#dict_zip = pickle.load(file)
#print(len(dict_zip['unstructured']))
#get_plot_size([dict_zip['unstructured'][x] for x in range(0,21, 2)], [x for x in range(0, 101, 10)])

#count parameters !=0
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    zeros = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        if parameter is not None:
            zeros_layer = torch.sum((parameter == 0).int()).item()
            zeros += zeros_layer
        else:
            print("Is None")
        param = parameter.numel()
        table.add_row([name, param - zeros_layer])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params-zeros}")
    return total_params-zeros


def parameters():
    dict_params = {k:[] for k in methods}
    path = './models/pruned'
    with tqdm(total=(10*3)+3, file=sys.stdout) as pbar:
        logging.disable(logging.CRITICAL)
        for filename in os.listdir(path):
            method = os.path.join(path, filename)
            for i in range(0, 105, 10):
                sys.stdout = open(os.devnull, 'w')
                model_path_pth = method+"/"+str(i)+'%_'+filename+'_pruned_model.pth'
                print(str(i)+'%_'+filename+'_pruned_model.pth')
                model = create_model()
                model.load(model_path_pth)
                tot_param = count_parameters(model)
                dict_params[filename].append(tot_param)
                sys.stdout = sys.__stdout__
                pbar.update(1)
    return dict_params

# parameters_dict = parameters()
# a_file = open("dict_parameters.pkl", "wb")
# pickle.dump(parameters_dict, a_file)
# a_file.close()

#calcolo differenze in percentuali parametri
def get_plot_parameters(value, x):
    width = 7
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ax.bar(x, value, width, color="blue", align='center')
    rects = ax.patches    
    for rect, label in zip(rects, value):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, str(format(label*1e-6, '.2f')), ha="center", va="bottom", color='black', fontsize=11)
            

    plt.xlabel('% Sparsity', fontweight='bold')
    plt.ylabel('# Parameters', fontweight='bold')
    plt.title('Trainable parameters pruned models')
    #plt.yticks(np.arange(0, max(value)+2e6, 1e6))
    #plt.show()
    plt.savefig('./images/Pruning_Unstructured_parameters.png')

#file = open("dict_parameters.pkl", "rb")
#dict_param = pickle.load(file)

#get_plot_parameters(dict_param['unstructured'], [x for x in range(0, 101, 10)])

def min_max_loss():
    file = open("dict_loss.pkl", "rb")
    dict_loss = pickle.load(file)
    dict_min_max = {k:[] for k in dict_loss.keys()}
    lst_max_comp = [0,0,0]
    lst_min_comp = [100,100,100]
    type_pruning = ''
    for type, lst_loss in dict_loss.items():
        if type_pruning == '':
            type_pruning = type
        else:
            if type_pruning!=type:
                lst_max_comp = [0,0,0]
                lst_min_comp = [100,100,100]
                type_pruning = type
        print()                
        for i in range(0, len(lst_loss)):
            for num in range (0,len(lst_loss[i])):
                if lst_loss[i][num] > lst_max_comp[num]:
                    lst_max_comp[num] = lst_loss[i][num]
                if lst_loss[i][num] < lst_min_comp[num]:
                    lst_min_comp[num] = lst_loss[i][num]
        dict_min_max[type].append(lst_max_comp)
        dict_min_max[type].append(lst_min_comp)
    print(dict_min_max)
                

#min_max_loss()

file = open("dict_loss.pkl", "rb")
dict_loss = pickle.load(file)
file = open("lst_perc_loss_global_Classification.pkl", "rb")
lst_loss = pickle.load(file)
print(lst_loss)