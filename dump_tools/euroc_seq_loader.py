""" For use in dumping single frame ground truths of EuRoc Dataset
Adapted from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/0caec9ed0f83cb65ba20678a805e501439d2bc25/data/kitti_raw_loader.py

You-Yi Jau, yjau@eng.ucsd.edu, 2019
Rui Zhu, rzhu@eng.ucsd.edu, 2019
"""

from __future__ import division
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.misc
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

from kitti_tools.utils_kitti import (
    load_velo_scan,
    rectify,
    read_calib_file,
    transform_from_rot_trans,
    scale_intrinsics,
    scale_P,
)
import dsac_tools.utils_misc as utils_misc

# from utils_good import *
from glob import glob
from dsac_tools.utils_misc import crop_or_pad_choice
from utils_kitti import load_as_float, load_as_array, load_sift, load_SP

import yaml

DEEPSFM_PATH = "/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm"
sys.path.append(DEEPSFM_PATH)
import torch
from models.model_wrap import PointTracker
from models.model_wrap import SuperPointFrontend_torch

from kitti_odo_loader import KittiOdoLoader
from kitti_odo_loader import *

coloredlogs.install(level="INFO", logger=logger)
# coloredlogs.install(level="DEBUG", logger=logger)

class euroc_seq_loader(KittiOdoLoader):
    def __init__(
        self,
        dataset_dir,
        img_height=375,
        img_width=1242,
        cam_ids=["00"],
        get_X=False,
        get_pose=False,
        get_sift=False,
        get_SP=False,
        sift_num=2000,
        if_BF_matcher=False,
        save_npy=True,
    ):
        # depth_size_ratio=1):
        # dir_path = Path(__file__).realpath().dirname()

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = cam_ids  # ['cam0/data']
        logging.info(f"cam id: {cam_ids}")
        # assert self.cam_ids == ['02'], 'Support left camera only!'
        self.cid_to_num = {"00": 0, "01": 1}

        self.debug = True # True
        if self.debug:
            self.train_seqs = ["MH_01_easy"]
            self.test_seqs = ["MH_01_easy"]
        else:
            self.train_seqs = [
                    "MH_01_easy",
                    "MH_02_easy",
                    "MH_04_difficult",
                    "V1_01_easy",
                    "V1_02_medium",
                    "V1_03_difficult",

                    ]
            self.test_seqs = [
                    "MH_02_easy",
                    "MH_05_difficult",
                    "V2_01_easy",
                    "V2_02_medium",
                    "V2_03_difficult",
                    ]

            # to_2darr = lambda x: np.array(x)
        self.test_seqs = np.char.add(self.test_seqs, "/mav0")
        self.train_seqs = np.char.add(self.train_seqs, "/mav0")

        # self.train_seqs = [4]
        # self.test_seqs = []
        # self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # self.test_seqs = []
        # self.map_to_raw = {
        #     "00": "2011_10_03_drive_0027",
        #     "01": "2011_10_03_drive_0042",
        #     "02": "2011_10_03_drive_0034",
        #     "03": "2011_09_26_drive_0067",
        #     "04": "2011_09_30_drive_0016",
        #     "05": "2011_09_30_drive_0018",
        #     "06": "2011_09_30_drive_0020",
        #     "07": "2011_09_30_drive_0027",
        #     "08": "2011_09_30_drive_0028",
        #     "09": "2011_09_30_drive_0033",
        #     "10": "2011_09_30_drive_0034",
        # }

        self.get_X = get_X
        self.get_pose = get_pose
        self.get_sift = get_sift
        self.get_SP = get_SP
        self.save_npy = save_npy
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
            # self.bf = cv2.BFMatcher(normType=cv2.NORM_L2)
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            # search_params = dict(checks = 50)
            # self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            # self.sift_matcher = self.bf if BF_matcher else self.flann

        self.scenes = {"train": [], "test": []}
        if self.get_SP:
            self.prapare_SP()
        # no need two functions
        self.collect_train_folders()
        self.collect_test_folders()

    def read_images_files_from_folder(self, drive_path, scene_data, folder="rgb"):
        # print(f"cid_num: {scene_data['cid_num']}")
        # img_dir = os.path.join(drive_path, "cam%d" % scene_data["cid_num"])
        # img_files = sorted(glob(img_dir + "/data/*.png"))
        print(f"drive_path: {drive_path}")
        ## given that we have matched time stamps
        arr = np.genfromtxt(
            f"{drive_path}/{folder}/data_f.txt", dtype="str"
        )  # [N, 2(time, path)]
        img_files = np.char.add(str(drive_path) + f"/{folder}/data/", arr[:, 1])
        img_files = [Path(f) for f in img_files]
        img_files = sorted(img_files)

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
        img_ori = scipy.misc.imread(img_file)
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
            img = scipy.misc.imresize(img_ori, (self.img_height, self.img_width))
            return img, (zoom_x, zoom_y), img_ori

    # def collect_scene_from_drive(self, drive_path):
    def collect_scene_from_drive(self, drive_path, split="train"):
        # adapt for Euroc dataset
        train_scenes = []
        logging.info("Gathering info for %s..." % drive_path)
        for c in self.cam_ids:
            scene_data = {
                "cid": c,
                "cid_num": self.cid_to_num[c],
                "dir": Path(drive_path),
                "rel_path": str(Path(drive_path).parent.name) + "_" + c,
            }
            # img_dir = os.path.join(drive_path, 'image_%d'%scene_data['cid_num'])
            # scene_data['img_files'] = sorted(glob(img_dir + '/*.png'))
            scene_data["img_files"] = self.read_images_files_from_folder(
                drive_path, scene_data, folder="cam0"
            )
            scene_data["N_frames"] = len(scene_data["img_files"])
            assert scene_data["N_frames"] != 0, "No file found for %s!" % drive_path
            scene_data["frame_ids"] = [
                "{:06d}".format(i) for i in range(scene_data["N_frames"])
            ]

            img_shape = None
            zoom_xy = None
            show_zoom_info = True
            # read images
            for idx in tqdm(range(scene_data["N_frames"])):
                img, zoom_xy, _ = self.load_image(scene_data, idx, show_zoom_info)
                # print(f"zoom_xy: {zoom_xy}")
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
            # print(img_shape)
            scene_data["calibs"] = {
                "im_shape": [img_shape[0], img_shape[1]],
                "zoom_xy": zoom_xy,
                "rescale": True if zoom_xy != (1.0, 1.0) else False,
            }

            # Get geo params from the RAW dataset calibs
            calib_file = os.path.join(
                drive_path, "cam%d" % scene_data["cid_num"], "sensor.yaml"
            )
            # calib_file = f"{scene_data['img_files'][0].str()}/../../sensor.yaml"
            P_rect_ori, cam2body_mat = self.get_P_rect(calib_file, scene_data["calibs"])
            P_rect_ori_dict = {c: P_rect_ori}
            intrinsics = P_rect_ori_dict[c][:, :3]
            logging.debug(f"intrinsics: {intrinsics}, cam2body_mat: {cam2body_mat}")
            calibs_rects = self.get_rect_cams(intrinsics, cam2body_mat[:3])  ##### need validation
            # calibs_rects = {"Rtl_gt": cam2body_mat}
            cam_2rect_mat = intrinsics

            # drive_in_raw = self.map_to_raw[drive_path[-2:]]
            # date = drive_in_raw[:10]
            # seq = drive_in_raw[-4:]
            # calib_path_in_raw = Path(self.dataset_dir)/'raw'/date
            # imu2velo_dict = read_calib_file(calib_path_in_raw/'calib_imu_to_velo.txt')
            # velo2cam_dict = read_calib_file(calib_path_in_raw/'calib_velo_to_cam.txt')
            # cam2cam_dict = read_calib_file(calib_path_in_raw/'calib_cam_to_cam.txt')
            # velo2cam_mat = transform_from_rot_trans(velo2cam_dict['R'], velo2cam_dict['T'])
            # imu2velo_mat = transform_from_rot_trans(imu2velo_dict['R'], imu2velo_dict['T'])
            # cam_2rect_mat = transform_from_rot_trans(cam2cam_dict['R_rect_00'], np.zeros(3))
            velo2cam_mat = None

            scene_data["calibs"].update(
                {
                    "K": intrinsics,
                    "P_rect_ori_dict": P_rect_ori_dict,
                    "cam_2rect": cam_2rect_mat,
                    "velo2cam": velo2cam_mat,
                    "cam2body_mat": cam2body_mat,
                }
            )
            scene_data["calibs"].update(calibs_rects)

            # Get pose
            poses = (
                np.genfromtxt(Path(drive_path) / "data_f.kitti".format(drive_path[-2:]))
                .astype(np.float32)
                .reshape(-1, 3, 4)
            )
            assert scene_data["N_frames"] == poses.shape[0], (
                "scene_data[N_frames]!=poses.shape[0], %d!=%d"
                % (scene_data["N_frames"], poses.shape[0])
            )
            scene_data["poses"] = poses

            # ground truth rt for camera
            scene_data["Rt_cam2_gt"] = scene_data["calibs"]["Rtl_gt"]

            train_scenes.append(scene_data)
        return train_scenes

    def get_P_rect(self, calib_file, calibs):
        # calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'

        calib_data = loadConfig(calib_file)
        height, width, K, D = load_intrinsics(calib_data)
        transformation_base_camera = load_extrinsics(calib_data)
        P_rect = np.concatenate((K, [[0], [0], [0]]), axis=1)
        if calibs["rescale"]:
            P_rect = scale_P(P_rect, calibs["zoom_xy"][0], calibs["zoom_xy"][1])
        return P_rect, transformation_base_camera

    @staticmethod
    def load_velo(scene_data, tgt_idx, calib_K=None):
        """
        create point clouds from depth image, return array of points 
        return:
            np [N, 3] (3d points)
        """
        logging.error(f"Not implemented error!! Turn off --with_X")
        return None


def load_intrinsics(calib_data):
    width, height = calib_data["resolution"]
    # cam_info.distortion_model = 'plumb_bob'
    D = np.array(calib_data["distortion_coefficients"])
    # cam_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    fu, fv, cu, cv = calib_data["intrinsics"]
    K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])

    return height, width, K, D


# parse camera calibration yaml file
def load_extrinsics(calib_data):
    # read homogeneous rotation and translation matrix
    transformation_base_camera = np.array(calib_data["T_BS"]["data"])
    transformation_base_camera = transformation_base_camera.reshape((4, 4))

    # compute projection matrix
    #  projection = np.zeros((3,4))
    #  projection[:,:-1] = K
    #  cam_info.P = projection.reshape(-1,).tolist()

    return transformation_base_camera


def loadConfig(filename):
    import yaml

    with open(filename, "r") as f:
        config = yaml.load(f)
    return config


# calib_file = '/data/euroc/mav0/cam0/sensor.yaml'
# calib_data = loadConfig(calib_file)

# intrinsics = load_intrinsics(calib_data)
# transformation_base_camera = load_extrinsics(calib_data)
# print(f"height, width, K, D = {intrinsics}")
# print(f"transformation_base_camera: {transformation_base_camera}")


if __name__ == "__main__":
    pass
