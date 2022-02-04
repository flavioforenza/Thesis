import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from vision.nn.mobilenet import MobileNetV1

from vision.ssd.ssd import SSD
from vision.ssd.predictor import Predictor
from vision.ssd.config import mobilenetv1_ssd_config as config
from models import MobileNetV1_Stud

def create_mobilenetv1_ssd(num_classes, alpha, is_test=False):
    #base_net = torch.load(path_student_distill).model
    base_net = MobileNetV1_Stud(num_classes, alpha).model  # disable dropout layer

    source_layer_indexes = [
        12,
        14,
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=int(1024*alpha), out_channels=int(256*alpha), kernel_size=1),
            ReLU(),
            Conv2d(in_channels=int(256*alpha), out_channels=int(512*alpha), kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=int(512*alpha), out_channels=int(128*alpha), kernel_size=1),
            ReLU(),
            Conv2d(in_channels=int(128*alpha), out_channels=int(256*alpha), kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
            ReLU(),
            Conv2d(in_channels=int(128*alpha), out_channels=int(256*alpha), kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
            ReLU(),
            Conv2d(in_channels=int(128*alpha), out_channels=int(256*alpha), kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=int(512*alpha), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(1024*alpha), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(512*alpha), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(256*alpha), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(256*alpha), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(256*alpha), out_channels=6 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=int(512*alpha), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(1024*alpha), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(512*alpha), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(256*alpha), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(256*alpha), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(256*alpha), out_channels=6 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv1_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
