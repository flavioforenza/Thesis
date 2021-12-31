#
# this script remaps the original Cityscapes class label ID's (range 0-33)
# to lie within a range that the segmentation networks support (21 classes)
#
# see below for the mapping of the original class ID's to the new class ID's,
# along with the class descriptions, some of which are combined from the originals.
#
# this script will overwrite the *_labelIds.png files from the gtCoarse/gtFine sets.
# to run it, launch these example commands for the desired train/train_extra/val sets:
#
#     $ python3 cityscapes_remap.py <path-to-cityscapes>/gtCoarse/train
#     $ python3 cityscapes_remap.py <path-to-cityscapes>/gtCoarse/val
#
import os
import copy
import argparse

from PIL import Image
from multiprocessing import Pool as ProcessPool
from cityscapes_utils import *
from torchvision import transforms

#
# map of existing class label ID's (range 0-33) to new ID's (range 0-21)
#
LABEL_MAP = [0,  # unlabeled
             1,  # ego vehicle
             0,  # rectification border
             0,  # out of roi
             2,  # static
             2,  # dynamic
             2,  # ground
             3,  # road
             4,  # sidewalk
             3,  # parking
             3,  # rail track
             5,  # building
             6,  # wall
             7,  # fence
             7,  # guard rail
             3,  # bridge
             3,  # tunnel
             8,  # pole
             8,  # polegroup
             9,  # traffic light
             10,  # traffic sign
             11,  # vegetation
             12,  # terrain
             13,  # sky
             14,  # person
             14,  # rider
             15,  # car
             16,  # truck
             17,  # bus
             16,  # caravan
             16,  # trailer
             18,  # train
             19,  # motorcycle
             20]  # bicycle

#
# new class label names, corresponding to remapped class ID's (range 0-21)
#
"""
void
ego_vehicle
ground
road
sidewalk
building
wall
fence
pole
traffic_light
traffic_sign
vegetation
terrain
sky
person
car
truck
bus
train
motorcycle
bicycle
"""


def remap_labels(filename):
    print(filename)
    img = Image.open(filename)

    for y in range(img.height):
        for x in range(img.width):
            org_label = img.getpixel((x, y))
            #new_value_pixel = LABEL_MAP[org_label]
            #if org_label != new_value_pixel:
            #    print("Label ", org_label, " changed with ", new_value_pixel)
            img.putpixel((x, y), LABEL_MAP[org_label])

    img.save(filename)


if __name__ == "__main__":

    # download CityScapes Dataset
    # path_dataset = '/home/flavio/thesis/jetson_nano/pytorch-segmentation/datasets/cityscapes/'
    # path_gtFine = '/home/flavio/thesis/jetson_nano/pytorch-segmentation/datasets/cityscapes/gtFine/train'
    # CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # transform = transforms.Compose([
    #    FilterAndRemapCityscapesCategories(CAT_LIST, torchvision.datasets.Cityscapes.classes),
    #    transforms
    # ])

    # dataset = torchvision.datasets.Cityscapes(path_dataset, split = 'train', mode='fine', target_type='semantic', transform=transforms, target_transform=transforms)

    parser = argparse.ArgumentParser(description='Remap Cityscapes Segmenation Labels')
    parser.add_argument('dir', metavar='DIR', help='path to data labels (e.g. gtCoarse/train, ect.)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    args = parser.parse_args()

    img_list = []

    for city in os.listdir(args.dir):
        img_dir = os.path.join(args.dir, city)

        for file_name in os.listdir(img_dir):
            if file_name.find("labelIds.png") == -1:
                continue
            img_list.append(os.path.join(img_dir, file_name))

    with ProcessPool(processes=args.workers) as pool:
        pool.map(remap_labels, img_list)


