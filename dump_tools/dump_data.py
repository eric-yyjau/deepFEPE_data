"""This is the script for data processing
it will call the classes for each dataset

Author: You-Yi Jau, Rui Zhu
Date: 2020/07/15
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.misc
import cv2
from glob import glob
import time

from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import argparse
from pebble import ProcessPool
import multiprocessing as mp

default_number_of_process = (
    1
)  # to prevent congestion; SIFT and matrix operations in recfity points already takes advantage of multi-cores
import time
import yaml

## split training set to train/val
def split_training_set(sample_name_lists, args_dump_root):
    sample_name_flat_list = [item for sublist in sample_name_lists for item in sublist]

    print("> Generating train val lists from %d samples..." % len(sample_name_flat_list))
    np.random.seed(8964)
    val_ratio = 0.2
    # to avoid data snooping, we will make two cameras of the same scene to fall in the same set, train or val
    # subdirs = (
    #     args_dump_root.dirs()
    # )  # e.g. Path('./data/kitti_dump/2011_09_30_drive_0034_sync_02')
    # canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs]) # e.g. '2011_09_28_drive_0039_sync_'
    with open(args_dump_root / "train.txt", "w") as tf:
        with open(args_dump_root / "val.txt", "w") as vf:
            for pr in tqdm(sample_name_flat_list):
                # corresponding_dirs = args_dump_root.dirs('{}*'.format(pr)) # e.g. [Path('./data/kitti_dump/2011_09_30_drive_0033_sync_03'), Path('./data/kitti_dump/2011_09_30_drive_0033_sync_02')]
                if np.random.random() < val_ratio:
                    # if pr[:2] in ['06', '07', '08', '09', '10']:
                    vf.write("{}\n".format(pr))
                else:
                    tf.write("{}\n".format(pr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foo")
    parser.add_argument(
        "--dataset_dir", type=str, default="/data/KITTI/raw_meta/", help="path to dataset"
    )
    parser.add_argument(
        "--dataloader_name", type=str, default="tum_seq_loader", help="number of thread to load data"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=default_number_of_process,
        help="current only support single thread",
    )
    parser.add_argument(
        "--cam_id", type=str, default="02", help="number of thread to load data"
    )
    parser.add_argument(
        "--img_height", type=int, default=376, help="number of thread to load data"
    )
    parser.add_argument(
        "--img_width", type=int, default=1241, help="number of thread to load data"
    )
    parser.add_argument(
        "--static_frames_file", type=str, default=None, help="static data file path"
    )
    parser.add_argument(
        "--test_scene_file", type=str, default=None, help="test data file path"
    )
    parser.add_argument("--dump", action="store_true", default=False)
    parser.add_argument(
        "--with_X",
        action="store_true",
        default=False,
        help="If available (e.g. with KITTI), will store visable rectified lidar points ground truth along with images, for validation",
    )
    parser.add_argument(
        "--with_pose",
        action="store_true",
        default=True,
        help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation",
    )
    parser.add_argument(
        "--with_sift",
        action="store_true",
        default=False,
        help="If available (e.g. with KITTI), will store SIFT points ground truth along with images, for validation",
    )
    parser.add_argument(
        "--with_SP",
        action="store_true",
        default=False,
        help="If available (e.g. with KITTI), will store SuperPoint points ground truth along with images, for validation",
    )
    parser.add_argument(
        "--dump_root", type=str, default="dump", help="Where to dump the data"
    )


    # args = parser.parse_args('--dump --with_X --with_pose --with_sift \
    #     --static_frames_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/static_frames.txt \
    #     --test_scene_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/test_scenes.txt \
    #     --dataset_dir /home/ruizhu/Documents/Datasets/kitti/raw \
    #     --dump_root /home/ruizhu/Documents/Datasets/kitti/corr_dump'.split())

    ## parameters
    args = parser.parse_args()
    print(args)

    delta_ijs = [1] # [1, 2, 3, 5, 8, 10]
    if args.with_sift:
        print(f"delta_ijs for sift: {delta_ijs}")
    splits = ['train', 'test']
    print(f"dump splits: {splits}")

    ## create dump folders
    args_dump_root = Path(args.dump_root)
    args_dump_root.mkdir(parents=True, exist_ok=True)

    ## output configs
    with open(os.path.join(args.dump_root, "config.yml"), "w") as f:
        yaml.dump(args, f, default_flow_style=False)
    logging.info(f"dump args to: {args.dump_root}/config.yml")

    ## get data loader
    def get_model(name, func):
        mod = __import__(name, fromlist=[func])
        return getattr(mod, name)
    # from tum_seq_loader import tum_seq_loader as seq_loader
    # from euroc_seq_loader import euroc_seq_loader as seq_loader
    # seq_loader_model = "euroc_seq_loader"
    seq_loader_model = args.dataloader_name
    seq_loader = get_model(seq_loader_model, seq_loader_model)
    logging.info(f"{seq_loader_model} is loaded!")


    data_loader = seq_loader(
        args.dataset_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        cam_ids=[args.cam_id],
        get_X=args.with_X,
        get_pose=args.with_pose,
        get_sift=args.with_sift,
        get_SP=args.with_SP,
        delta_ijs = delta_ijs 
    )

    print(f"data_loader: {data_loader}")
    n_scenes = {
        "train": len(data_loader.scenes["train"]),
        "test": len(data_loader.scenes["test"]),
    }
    print(
        "Found %d potential train scenes, and %d test scenes."
        % (n_scenes["train"], n_scenes["test"])
    )

    print(f"one scene: {data_loader.scenes['train'][0]}")


    sample_name_lists = []
    # splits = ['train']
    for split in splits:
        print("> Retrieving frames for %s..." % split)
        seconds = time.time()

        def dump_scenes_from_drive(args, split, drive_path):
            # scene = data_loader.collect_scene_from_drive(drive_path)
            sample_name_list = data_loader.dump_drive(
                # args, drive_path, split=split, scene_data=None, skip_dumping=True
                args, drive_path, split=split, scene_data=None,
            )
            return sample_name_list

        if args.num_threads == 1:
            for drive_path in tqdm(data_loader.scenes[split]):
                print("Dumping ", drive_path)
                try:
                    sample_name_list = dump_scenes_from_drive(args, split, drive_path)
                except:
                    logging.warning(f"problem occurs when dumping {drive_path}, skip this sequence")
                    raise
                    pass

                if split == "train":
                    sample_name_lists.append(sample_name_list)
                elif split == "test":
                    ## write to test.txt
                    with open(args_dump_root / "test.txt", "a") as vf:
                        sample_name_flat_list = [sublist for sublist in sample_name_list]
                        for pr in tqdm(sample_name_flat_list):
                            vf.write("{}\n".format(pr))
        else: 
            raise NotImplementedError

        if split == "train":
            split_training_set(sample_name_lists, args_dump_root)

        print("<<< Finished dump %s scenes. " % split, time.time() - seconds)

