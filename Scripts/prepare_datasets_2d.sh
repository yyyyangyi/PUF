#!/bin/bash
# The 3RScan dataset should be downloaded and extracted to the ../Datasets/3RScan directory
set -euxo pipefail


# get 3RScan metadata
# cp /home/yangyi/tmp/data/3RScan/data/3RScan/3RScan.json /home/yangyi/tmp/data/3RScan/data/3DSSG22D/
# cp /home/yangyi/tmp/data/3RScan/data/3RScan/objects.json /home/yangyi/tmp/data/3RScan/data/3DSSG22D/

# cd /home/yangyi/tmp/data/3RScan/data/3DSSG22D/
# wget http://campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset.zip
# unzip 3DSSG_subset.zip
# rm 3DSSG_subset.zip
# cd /home/yangyi/tmp/code/FROSS/Scripts

# # copy the 3DSSG to ScanNet categories mapping, which can be found here: https://docs.google.com/spreadsheets/d/1eRTJ2M9OHz7ypXfYD-KTR1AIT-CrVLmhJf8mxgVZWnI/edit?usp=sharing
# cp files/3dssg_to_scannet.json /home/yangyi/tmp/data/3RScan/data/3DSSG22D/3DSSG_subset/3dssg_to_scannet.json
# cp /home/yangyi/tmp/data/3RScan0/3DSSG_2D/3dssg_to_scannet_2d.json /home/yangyi/tmp/data/3RScan/data/3DSSG22D/3DSSG_subset/3dssg_to_scannet_2d.json

# # copy our train/val split since the original mixed up reference scans and rescans
# cp files/train_scans.txt /home/yangyi/tmp/data/3RScan/data/3DSSG22D/3DSSG_subset/train_scans.txt
# cp files/validation_scans.txt /home/yangyi/tmp/data/3RScan/data/3DSSG22D/3DSSG_subset/validation_scans.txt

# # We use the validation split in 3RScan as the test split
# wget https://campar.in.tum.de/public_datasets/3RScan/val_scans.txt -O /home/yangyi/tmp/data/3RScan/data/3DSSG22D/3DSSG_subset/test_scans.txt

# extract scannet classes annotations from 3RScan dataset
# python dataset/relationship2scannet.py --path /home/yangyi/tmp/data/3RScan/data/3DSSG22D

# extract 2D bounding boxes from 3RScan dataset
python dataset/boxes2coco.py --path /home/yangyi/tmp/data/3RScan/data/3DSSG22D --label_categories scannet

# sanity checks
# python tools/show_bbox.py --path /home/yangyi/tmp/data/3RScan/data/3DSSG22D
# python tools/show_sg_labels.py --path /home/yangyi/tmp/data/3RScan/data/3DSSG22D
