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
from kitti_odo_loader import (
    dump_sift_match_idx,
    get_sift_match_idx_pair,
    dump_SP_match_idx,
    get_SP_match_idx_pair,
    read_odo_calib_file,
)


class tum_seq_loader(KittiOdoLoader):
    def __init__(
        self,
        dataset_dir,
        img_height=375,
        img_width=1242,
        cam_ids=["00"],  # no usage in TUM
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
        self.cam_ids = ["00"]  # no use in TUM
        # assert self.cam_ids == ['02'], 'Support left camera only!'
        self.cid_to_num = {"00": 0, "01": 1, "02": 2, "03": 3}

        self.debug = False
        if self.debug:
            coloredlogs.install(level="DEBUG", logger=logger) # original info

        if self.debug:
            ## small dataset for debuggin
            self.train_seqs = ["rgbd_dataset_freiburg1_xyz"]
            self.test_seqs = ["rgbd_dataset_freiburg1_xyz"]
        else:
            ## dataset names
            self.train_seqs = [
                "rgbd_dataset_freiburg1_desk",
                "rgbd_dataset_freiburg1_room",
                "rgbd_dataset_freiburg2_desk",
                "rgbd_dataset_freiburg3_long_office_household",
            ]
            self.test_seqs = [
                "rgbd_dataset_freiburg1_desk2",
                "rgbd_dataset_freiburg2_xyz",
                "rgbd_dataset_freiburg3_nostructure_texture_far",
            ]

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
        print(f"drive_path: {drive_path}")
        ## given that we have matched time stamps
        arr = np.genfromtxt(f'{drive_path}/{folder}_filter.txt',dtype='str') # [N, 2(time, path)]
        img_files = np.char.add(str(drive_path)+'/',arr[:,1])
        img_files = [Path(f) for f in img_files]
        img_files = sorted(img_files)

        ## no time stamps
        # img_dir = os.path.join(drive_path, "")
        # img_files = sorted(glob(img_dir + f"/{folder}/*.png"))
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

    def get_calib_file_from_folder(self, foldername):
        cid = 1
        cam_name = "freiburg"
        for i in range(1, 4):
            if f"{cam_name}{i}" in str(foldername):
                cid = i
        calib_file = f"{self.dataset_dir}/tum/TUM{cid}.yaml"
        return calib_file

    # def collect_scene_from_drive(self, drive_path):
    def collect_scene_from_drive(self, drive_path, split="train"):
        # adapt for Euroc dataset
        train_scenes = []
        logging.info("Gathering info for %s..." % drive_path)
        for c in self.cam_ids:
            scene_data = {
                "cid": "00",
                "cid_num": 0,
                "dir": Path(drive_path),
                "rel_path": Path(drive_path).name + "_" + "00",
            }
            # img_dir = os.path.join(drive_path, 'image_%d'%scene_data['cid_num'])
            # scene_data['img_files'] = sorted(glob(img_dir + '/*.png'))
            scene_data["img_files"] = self.read_images_files_from_folder(
                drive_path, scene_data, folder="rgb"
            )
            scene_data["depth_files"] = self.read_images_files_from_folder(
                drive_path, scene_data, folder="depth"
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
            # calib_file = os.path.join("tum/TUM1.yaml")
            # calib_file = os.path.join("/data/tum/calib/TUM1.yaml")
            calib_file = os.path.join(self.get_calib_file_from_folder(drive_path))
            logging.info(f"calibration file: {calib_file}")
            # calib_file = f"{scene_data['img_files'][0].str()}/../../sensor.yaml"
            P_rect_noScale, P_rect_scale = self.get_P_rect(
                calib_file, scene_data["calibs"]
            )
            P_rect_ori_dict = {c: P_rect_scale}
            intrinsics = P_rect_ori_dict[c][:, :3]
            logging.debug(f"intrinsics: {intrinsics}")
            # calibs_rects = self.get_rect_cams(intrinsics, P_rect_ori_dict[c])
            calibs_rects = {"Rtl_gt": np.eye(4)}  # only one camera, no extrinsics
            cam_2rect_mat = np.eye(4)  # extrinsics for cam2

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

            # Get pose
            gt_kitti_file = "groundtruth_filter.kitti"
            # if not (Path(drive_path) / gt_kitti_file).exists():
            #     import subprocess
            #     gt_file = "groundtruth_filter.txt"
            #     assert (Path(drive_path) / gt_file).exists()
            #     # process files
            #     logging.info(f"generate kitti format gt pose: {drive_path}")
            #     subprocess.run(f"evo_traj tum {str(Path(drive_path)/gt_file)} --save_as_kitti", shell=True, check=True) # https://github.com/MichaelGrupp/evo

            assert (
                Path(drive_path) / gt_kitti_file
            ).exists(), "kitti style of gt pose file not found, please run 'python process_poses.py --dataset_dir DATASET_DIR"
            poses = (
                np.genfromtxt(Path(drive_path) / gt_kitti_file)
                .astype(np.float32)
                .reshape(-1, 3, 4)
            )

            # print(f"poses before: {poses[:10]}")
            # ## invert camera poses of world coord to poses of camera coord
            # poses = np.array([np.linalg.inv(utils_misc.Rt_pad(pose))[:3] for pose in poses])
            # print(f"poses after: {poses[:10]}")

            assert scene_data["N_frames"] == poses.shape[0], (
                "scene_data[N_frames]!=poses.shape[0], %d!=%d"
                % (scene_data["N_frames"], poses.shape[0])
            )
            scene_data["poses"] = poses

            # extrinsic matrix for cameraN to this camera
            scene_data["Rt_cam2_gt"] = scene_data["calibs"]["Rtl_gt"]
            logging.debug(f'scene_data["Rt_cam2_gt"]: {scene_data["Rt_cam2_gt"]}')

            train_scenes.append(scene_data)
        return train_scenes

    def get_P_rect(self, calib_file, calibs):
        # width, height = calib_data['resolution']
        # cam_info.distortion_model = 'plumb_bob'
        # D = np.array(calib_data['distortion_coefficients'])
        # cam_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        calib_data = loadConfig(calib_file)
        fu, fv, cu, cv = (
            calib_data["Camera.fx"],
            calib_data["Camera.fy"],
            calib_data["Camera.cx"],
            calib_data["Camera.cy"],
        )
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])

        P_rect_ori = np.concatenate((K, [[0], [0], [0]]), axis=1)
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

        # def get_point_cloud_from_images(color_file, depth_file):
        #     """
        #     will cause crashes!!!
        #     """
        #     import open3d as o3d # import open3d before torch to avoid crashes
        #     depth_raw = o3d.io.read_image(depth_file)
        #     color_raw = o3d.io.read_image(color_file)
        #     rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
        #         color_raw, depth_raw)
        #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #         rgbd_image,
        #         o3d.camera.PinholeCameraIntrinsic(
        #             o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        #     # Flip it, otherwise the pointcloud will be upside down
        #     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #     xyz_points = np.asarray(pcd.points)
        #     return xyz_points

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


# calib_file = '/data/euroc/mav0/cam0/sensor.yaml'
# calib_data = loadConfig(calib_file)

# intrinsics = load_intrinsics(calib_data)
# transformation_base_camera = load_extrinsics(calib_data)
# print(f"height, width, K, D = {intrinsics}")
# print(f"transformation_base_camera: {transformation_base_camera}")


if __name__ == "__main__":
    pass
