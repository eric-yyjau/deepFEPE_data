""" For use in dumping single frame ground truths of Apollo training Dataset
Adapted from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/0caec9ed0f83cb65ba20678a805e501439d2bc25/data/kitti_raw_loader.py

Authors:
    You-Yi Jau, yjau@eng.ucsd.edu, 2020
    Rui Zhu, rzhu@eng.ucsd.edu, 2019
Date: 
    2020/07/15
"""

from __future__ import division
import numpy as np
from pathlib import Path
from tqdm import tqdm
# import scipy.misc
from collections import Counter
from pebble import ProcessPool
import multiprocessing as mp

ratio_CPU = 0.8
default_number_of_process = int(ratio_CPU * mp.cpu_count())

import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import traceback

import coloredlogs, logging

logging.basicConfig()
logger = logging.getLogger()
coloredlogs.install(level="INFO", logger=logger)

import cv2

from dump_tools.utils_kitti import (
    scale_P,
)
# import dsac_tools.utils_misc as utils_misc
# from dsac_tools.utils_misc import crop_or_pad_choice

# from utils_good import *
from glob import glob
# from utils_kitti import load_as_float, load_as_array, load_sift, load_SP

import yaml

# DEEPSFM_PATH = "/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm"
# sys.path.append(DEEPSFM_PATH)
import torch

# from kitti_odo_loader import KittiOdoLoader
from kitti_seq_loader import kitti_seq_loader
# from kitti_odo_loader import (
#     dump_sift_match_idx,
#     get_sift_match_idx_pair,
#     dump_SP_match_idx,
#     get_SP_match_idx_pair,
#     read_odo_calib_file,
# )

## apollo specific
from apollo.eval_pose import eval_pose
# from apollo_seq_loader import apollo_seq_loader


class apollo_train_loader(kitti_seq_loader):
    def __init__(
        self,
        dataset_dir,
        img_height=2710,
        img_width=3384,
        cam_ids=["5"],  # no usage in TUM
        get_X=False,
        get_pose=False,
        get_sift=False,
        get_SP=False,
        sift_num=2000,
        if_BF_matcher=False,
        save_npy=True,
        delta_ijs=[1]
    ):
        # original size: (H2710, W3384)
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = cam_ids[0]  # ["1"]  # no use in TUM
        logging.info(f"cam_id: {cam_ids}")
        assert self.cam_ids in ["5"], "Support left camera only!"
        self.cid_to_num = {"1": 1, "2": 2, "5": 5, "6": 6}

        ## testing set maps to val.txt
        self.split_mapping = {"train": "train", "test": "val"}

        self.debug = False
        if self.debug:
            coloredlogs.install(level="DEBUG", logger=logger)  # original info

        flat_list = lambda x: [item for sublist in x for item in sublist]
        if self.debug: # you can edit the split txt
            ## small dataset for debuggin
            split_folder = "split_small"
            self.train_seqs = ["Road11"]  # the folders after the dataset_dir
            self.test_seqs = ["Road11"]

        else:
            split_folder = "split"
            ## dataset names
            # self.train_seqs = ["Road16"]  # the folders after the dataset_dir
            # self.test_seqs = ["Road16"]
            self.train_seqs = ["Road11"]  # the folders after the dataset_dir
            self.test_seqs = ["Road11"]

        ## prepare training seqs
        self.train_rel_records = [
            np.genfromtxt(
                f"{dataset_dir}/{seq}/{split_folder}/{self.split_mapping['train']}.txt",
                dtype="str",
            )
            for seq in self.train_seqs
        ]
        print(f"self.train_rel_records: {self.train_rel_records}")
        self.train_rel_records = flat_list(self.train_rel_records)
        ## prepare testing seqs
        self.test_rel_records = [
            np.genfromtxt(
                f"{dataset_dir}/{seq}/{split_folder}/{self.split_mapping['test']}.txt",
                dtype="str",
            )
            for seq in self.test_seqs
        ]
        self.test_rel_records = flat_list(self.test_rel_records)
        logging.info(f"train_seqs: {self.train_seqs}, test_seqs: {self.test_seqs}, train_rel_records: {self.train_rel_records}, test_rel_records: {self.test_rel_records}")

        self.get_X = get_X
        self.get_pose = get_pose
        self.get_sift = get_sift
        self.get_SP = get_SP
        self.save_npy = save_npy
        self.delta_ijs = delta_ijs
        if self.save_npy:
            logging.info("+++ Dumping as npy")
        else:
            logging.info("+++ Dumping as h5")
        if self.get_sift:
            self.sift_num = sift_num
            self.if_BF_matcher = if_BF_matcher
            self.sift = cv2.xfeatures2d.SIFT_create(
                nfeatures=self.sift_num, contrastThreshold=1e-5
            )

        self.scenes = {
            "train": [],
            "test": [],
            "train_rel_records": self.train_rel_records,
            "test_rel_records": self.test_rel_records,
        }
        if self.get_SP:
            self.prapare_SP()
        # no need two functions
        self.collect_train_folders()
        self.collect_test_folders()

    @staticmethod
    def filter_list(list, select_word=""):
        return [l for l in list if select_word in l]

    def read_images_files_from_folder(
        self, drive_path, seq_folders, file="train.txt", cam_id=1
    ):
        """
        seq_folders: list of relative paths from drive_path to the image folders

        """
        flat_list = lambda x: [item for sublist in x for item in sublist]
        print(f"drive_path: {drive_path}")
        ## given that we have matched time stamps
        # arr = np.genfromtxt(f'{drive_path}/{file}',dtype='str') # [N, 1(path)]
        img_folders = np.char.add(str(drive_path) + "/image/", seq_folders)
        img_folders = np.char.add(img_folders, f"/Camera {cam_id}")
        logging.info(f"img_folders: {img_folders}")
        img_files = [glob(f"{folder}/*.jpg") for folder in img_folders]
        img_files = flat_list(img_files)
        img_files = sorted(img_files)

        ## no time stamps
        print(f"img_files: {img_files[0]}")
        return img_files

    def collect_train_folders(self):
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, seq)
            self.scenes["train"].append(seq_dir)

    def collect_test_folders(self):
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, seq)
            self.scenes["test"].append(seq_dir)

    def load_image(self, scene_data, tgt_idx, show_zoom_info=True):
        # use different image filename
        img_file = Path(scene_data["img_files"][tgt_idx])
        if not img_file.is_file():
            logging.warning("Image %s not found!" % img_file)
            return None, None, None
        img_ori = cv2.imread(str(img_file))
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        if [self.img_height, self.img_width] == [img_ori.shape[0], img_ori.shape[1]]:
            return img_ori, (1.0, 1.0), img_ori
        else:
            zoom_y = self.img_height / img_ori.shape[0]
            zoom_x = self.img_width / img_ori.shape[1]
            if show_zoom_info:
                logging.warning(
                    "[%s] Zooming the image (H%d, W%d) with zoom_yH=%f, zoom_xW=%f to (H%d, W%d)."
                    % (
                        img_file,
                        img_ori.shape[0],
                        img_ori.shape[1],
                        zoom_y,
                        zoom_x,
                        self.img_height,
                        self.img_width,
                    )
                )
            # img = scipy.misc.imresize(img_ori, (self.img_height, self.img_width))
            img = cv2.resize(img_ori, (self.img_width, self.img_height))
            return img, (zoom_x, zoom_y), img_ori

    def get_calib_file_from_folder(self, foldername):
        """ get camera intrinsics file
        """
        for i in self.cam_ids:
            if i == 1 or i == 2:
                calib_file = f"{foldername}/camera_params/Camera_{i}.cam"
            else:
                calib_file = f"{foldername}/camera_params/Camera\ {i}.cam"
        return calib_file

    @staticmethod
    def get_pose_to_dict(pose_path, cam_id=1):
        """ get ground truth camera poses
        """
        print(f"pose_path: {pose_path}")
        eval_agent = eval_pose({})
        if cam_id == 1 or cam_id == 2:
            pose_files = glob(f"{pose_path}/**/Camera_{cam_id}.txt")
        else:
            pose_files = glob(f"{pose_path}/**/Camera {cam_id}.txt")
        logging.debug(f"pose: {pose_files[0]}")
        # calib_data = []
        pose_dict = {}
        for i, f in enumerate(pose_files):
            print(f"file: {f}")
            data = eval_agent.load_pose_file(f, sep="  ")
            # print(f"data: {data}")
            pose_dict.update(data)
        return pose_dict


    @staticmethod
    def get_pose_from_pose_dict(pose_dict, img_files):
        from apollo.utils import euler_angles_to_rotation_matrix
        """
        input:
            pose: list of poses(np) [[[roll,pitch,yaw,x,y,z]], ...]
        """
        # poses = [pose_dict[Path(f).name] for f in img_files]
        poses = []
        for f in img_files:
            pose = pose_dict[Path(f).name].flatten()
            # print(f"pose: {pose}")
            rot = euler_angles_to_rotation_matrix(pose[:3])
            trans = pose[3:6]
            pose_mat = np.concatenate((rot, trans.reshape(-1, 1)), axis=1)
            poses.append(pose_mat.flatten())
        return np.array(poses)

    def collect_scene_from_drive(self, drive_path, split="train", skip_dumping=False):
        # adapt for Euroc dataset
        train_scenes = []
        split_mapping = self.split_mapping
        # split_mapping = {'train': 'train', 'test': 'val'}
        logging.info(f"Gathering {split} for {drive_path} ...")
        for c in self.cam_ids:
            scene_data = {
                "cid": c,
                "cid_num": self.cid_to_num[c],
                "dir": Path(drive_path),
                "rel_path": f"{split}/{Path(drive_path).name}_{c}",
            }
            # img_dir = os.path.join(drive_path, 'image_%d'%scene_data['cid_num'])
            # scene_data['img_files'] = sorted(glob(img_dir + '/*.png'))
            split_folder = "trainval_split" if self.debug else "split"
            scene_data["img_files"] = self.read_images_files_from_folder(
                self.scenes[f"{split}"][0],
                self.scenes[f"{split}_rel_records"],
                file=f"",
                cam_id=c,
            )
            # scene_data["depth_files"] = self.read_images_files_from_folder(
            #     drive_path, scene_data, folder="depth"
            # )
            scene_data["N_frames"] = len(scene_data["img_files"])
            assert scene_data["N_frames"] != 0, "No file found for %s!" % drive_path
            scene_data["frame_ids"] = [
                "{:06d}".format(i) for i in range(scene_data["N_frames"])
            ]

            ## Get gt poses
            pose_dict = self.get_pose_to_dict(
                f"{str(drive_path)}/pose/{Path(self.scenes[f'{split}_rel_records'][0]).parent.name}",
                cam_id=c,
            )
            poses = self.get_pose_from_pose_dict(pose_dict, scene_data["img_files"])
            assert scene_data["N_frames"] == poses.shape[0], (
                "scene_data[N_frames]!=poses.shape[0], %d!=%d"
                % (scene_data["N_frames"], poses.shape[0])
            )
            logging.info(f"N_frames: {scene_data['N_frames']}, n_poses: {poses.shape[0]}")
            scene_data["poses"] = poses

            ## read images
            img_shape = None
            zoom_xy = None
            show_zoom_info = True
            if not skip_dumping:
                for idx in tqdm(range(scene_data["N_frames"])):
                    img, zoom_xy, img_ori = self.load_image(scene_data, idx, show_zoom_info)
                    # print(f"zoom_xy: {zoom_xy}")
                    if idx % 100 == 0:
                        logging.info(
                            f"img: {img.shape}, img_ori: {img_ori.shape}, zoom_xy: {zoom_xy}"
                        )
                    show_zoom_info = False
                    if img is None and idx == 0:
                        logging.warning("0 images in %s. Skipped." % drive_path)
                        return []
                    else:
                        if img_shape is not None:
                            assert img_shape == img.shape, (
                                "Inconsistent image shape in seq %s!" % drive_path
                            )
                        else:
                            img_shape = img.shape
            else:
                logging.warning(f"skip dumping images!!")
                img_shape = [1,1,3]
                img_ori = np.zeros((1,1,3))  ## dummy image
                zoom_xy = [1,1]
                
            logging.debug(f"img_shape: {img_shape}")
            scene_data["calibs"] = {
                "im_shape": [img_shape[0], img_shape[1]],
                "zoom_xy": zoom_xy,
                "rescale": True if zoom_xy != (1.0, 1.0) else False,
            }

            # Get geo params from the RAW dataset calibs
            if c == "1" or c == "2": # for kitti
                calib_file = os.path.join(self.get_calib_file_from_folder(drive_path))
                logging.info(f"calibration file: {calib_file}")
                P_rect_noScale = self.get_cam_cali(calib_file)
            elif c == "5" or c == "6": # for apollo
                from apollo.data import ApolloScape
                from apollo.utils import intrinsic_vec_to_mat

                apo_data = ApolloScape()
                intr_vect = apo_data.get_intrinsic(
                    image_name=False, camera_name=f"Camera_{c}"
                )
                K = intrinsic_vec_to_mat(intr_vect, img_ori.shape)
                P_rect_noScale = np.concatenate((K, [[0], [0], [0]]), axis=1)

            logging.info(f"P_rect_noScale: {P_rect_noScale}")

            P_rect_noScale, P_rect_scale = self.get_P_rect(
                P_rect_noScale, scene_data["calibs"]
            )
            P_rect_ori_dict = {c: P_rect_scale}
            intrinsics = P_rect_ori_dict[c][:, :3]
            logging.debug(f"intrinsics: {intrinsics}")
            # calibs_rects = self.get_rect_cams(intrinsics, P_rect_ori_dict[c])
            calibs_rects = {"Rtl_gt": np.eye(4)}  # only one camera, no extrinsics
            ## dummy matrices
            cam_2rect_mat = np.eye(4)  # extrinsics for cam2
            velo2cam_mat = np.eye(4)
            cam2body_mat = np.eye(3)

            scene_data["calibs"].update(
                {
                    "K": intrinsics,
                    "P_rect_ori_dict": P_rect_ori_dict,
                    "P_rect_noScale": P_rect_noScale,  # add for read and process 3d points
                    "cam_2rect": cam_2rect_mat,
                    "velo2cam": velo2cam_mat,
                    "cam2body_mat": cam2body_mat,
                }
            )
            scene_data["calibs"].update(calibs_rects)

            # extrinsic matrix for cameraN to this camera
            scene_data["Rt_cam2_gt"] = scene_data["calibs"]["Rtl_gt"]
            logging.debug(f'scene_data["Rt_cam2_gt"]: {scene_data["Rt_cam2_gt"]}')

            train_scenes.append(scene_data)
        return train_scenes

    def get_cam_cali(self, calib_file):
        """ get calibration matrix
        """
        calib_data = np.genfromtxt(calib_file, delimiter="=", comments="[", dtype=str)
        fu, fv, cu, cv = (
            float(calib_data[3, 1]),
            float(calib_data[4, 1]),
            float(calib_data[5, 1]),
            float(calib_data[6, 1]),
        )
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])

        P_rect_ori = np.concatenate((K, [[0], [0], [0]]), axis=1)
        return P_rect_ori

    def get_P_rect(self, P_rect_ori, calibs):
        """ rescale the camera calibration matrix
        """
        # rescale the camera matrix

        if calibs["rescale"]:
            P_rect_scale = scale_P(
                P_rect_ori, calibs["zoom_xy"][0], calibs["zoom_xy"][1]
            )
        else:
            P_rect_scale = P_rect_ori

        return P_rect_ori, P_rect_scale

    @staticmethod
    def load_velo(scene_data, tgt_idx, calib_K=None):
        """
        create point clouds from depth image, return array of points 
        return:
            np [N, 3] (3d points)
        """
        depth_file = scene_data["depth_files"][tgt_idx]
        color_file = scene_data["img_files"][tgt_idx]

        def get_point_cloud_from_images(color_file, depth_file, calib_K=None):
            from PIL import Image

            depth = Image.open(depth_file)
            rgb = Image.open(color_file)
            points = []

            ## parameters
            if calib_K is None:
                focalLength = 525.0
                centerX = 319.5
                centerY = 239.5
            else:
                focalLength = (calib_K[0, 0] + calib_K[1, 1]) / 2
                centerX = calib_K[0, 2]
                centerY = calib_K[1, 2]
                logging.debug(
                    f"get calibration matrix for retrieving points: focalLength = {focalLength}, centerX = {centerX}, centerY = {centerY}"
                )

            scalingFactor = 5000.0

            for v in range(rgb.size[1]):
                for u in range(rgb.size[0]):
                    color = rgb.getpixel((u, v))
                    Z = depth.getpixel((u, v)) / scalingFactor
                    if Z == 0:
                        continue
                    X = (u - centerX) * Z / focalLength
                    Y = (v - centerY) * Z / focalLength
                    #             points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
                    points.append([X, Y, Z])

            logging.debug(f"points: {points[:3]}")
            return np.array(points)
            pass

        ###
        if Path(color_file).is_file() is False or Path(depth_file).is_file() is False:
            logging.warning(
                f"color file {color_file} or depth file {depth_file} not found!"
            )
            return None

        xyz_points = get_point_cloud_from_images(
            color_file, depth_file, calib_K=calib_K
        )
        # xyz_points = np.ones((10,3)) ######!!!
        logging.debug(f"xyz: {xyz_points[0]}, {xyz_points.shape}")

        return xyz_points


def loadConfig(filename):
    import yaml

    with open(filename, "r") as f:
        config = yaml.load(f)
    return config



if __name__ == "__main__":
    from apollo_train_loader import apollo_train_loader as seq_loader

    # test pose
    # from apollo_train_loader import get_pose_to_dict
    dataset_dir = "/newfoundland/yyjau/apollo/train_seq_1/"
    data_loader = seq_loader(dataset_dir)
    pose_path = "/newfoundland/yyjau/apollo/train_seq_1/Road11/pose/GZ20180310B"
    pose_dict = data_loader.get_pose_to_dict(pose_path, cam_id=5)

    pass
