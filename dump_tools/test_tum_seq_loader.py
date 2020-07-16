# NAME: loader.py
# DESCRIPTION: data loader for raw kitti data

import os
import sys
# sys.path.append('/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm_ori/FME')

import numpy as np
import scipy.misc
import os
import cv2
from glob import glob
import time

from pathlib import Path
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from torch.utils.data import Dataset

# for test
# from config import get_config
# config, unparsed = get_config()

import argparse
from pebble import ProcessPool
import multiprocessing as mp
# ratio_CPU = 0.5
# default_number_of_process = int(ratio_CPU * mp.cpu_count())
default_number_of_process = 1 # to prevent congestion; SIFT and matrix operations in recfity points already takes advantage of multi-cores
import time

import sys; sys.argv=['']; del sys # solve the error for jupyter notebook

parser = argparse.ArgumentParser(description='Foo')
parser.add_argument("--dataset_dir", type=str, default="/data/KITTI/raw_meta/", help="path to dataset")
parser.add_argument("--num_threads", type=int, default=default_number_of_process, help="number of thread to load data")
parser.add_argument("--cam_id", type=str, default='02', help="number of thread to load data")
parser.add_argument("--img_height", type=int, default=376, help="number of thread to load data")
parser.add_argument("--img_width", type=int, default=1241, help="number of thread to load data")
parser.add_argument("--static_frames_file", type=str, default=None, help="static data file path")
parser.add_argument("--test_scene_file", type=str, default=None, help="test data file path")
parser.add_argument('--dump', action='store_true', default=False)
parser.add_argument("--with_X", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store visable rectified lidar points ground truth along with images, for validation")
parser.add_argument("--with_pose", action='store_true', default=True,
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--with_sift", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store SIFT points ground truth along with images, for validation")
parser.add_argument("--with_SP", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store SuperPoint points ground truth along with images, for validation")
parser.add_argument("--dump_root", type=str, default='dump', help="Where to dump the data")

# args = parser.parse_args('--dump --with_X --with_pose --with_sift \
#     --static_frames_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/static_frames.txt \
#     --test_scene_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/test_scenes.txt \
#     --dataset_dir /home/ruizhu/Documents/Datasets/kitti/raw \

# no_X
args = parser.parse_args("--dump --dataset_dir /data/tum --with_pose  \
                        --with_sift --dump_root /data/tum_test/ \
                        --num_threads=1  --cam_id 00".split())
print(args)

# %reload_ext autoreload
# %autoreload 2

from tum_seq_loader import tum_seq_loader as seq_loader
assert args.cam_id in ['00', '02'], 'Only supported left greyscale/color cameras (cam 00 or 02)!'
data_loader = seq_loader(args.dataset_dir,
                             img_height=args.img_height,
                             img_width=args.img_width,
                             cam_ids=[args.cam_id],
                             get_X=args.with_X,
                             get_pose=args.with_pose,
                             get_sift=args.with_sift,
                             get_SP=args.with_SP)

print(f"data_loader: {data_loader}")
n_scenes = {'train': len(data_loader.scenes['train']), 'test': len(data_loader.scenes['test'])}
print('Found %d potential train scenes, and %d test scenes.'%(n_scenes['train'], n_scenes['test']))

print(f"one scene: {data_loader.scenes['train'][0]}")

# dump scenes
drive_path = data_loader.scenes['train'][0]
split = 'train'
data_loader.dump_drive(args, drive_path, split=split, scene_data=None)
