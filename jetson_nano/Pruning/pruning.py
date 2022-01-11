
import torch
import torch.nn.utils.prune as prune
import sys
import logging
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Using CUDA...")

def prune_net(val_pruned):
    create_net = create_mobilenetv1_ssd

    # num_classe == num_labels in models/labeles.txt
    num_classes = 11
    net = create_net(num_classes)

    net.eval()

    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            #print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement())))
            #print("####### BEFORE #######")
            print("Name:", name) 
            print("Module", module)
            #print(list(module.named_parameters()))
            #print(module.weight)
            prune.l1_unstructured(module, name='weight', amount=val_pruned/100)
            #print("####### AFTER #######")
            #print(module.weight)
            print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement())))

    return net

#Save model prunned
pruned_amount = 20
net = prune_net(pruned_amount)
net.save('./models/' + str(pruned_amount) + '%_pruned_model_100.pth')