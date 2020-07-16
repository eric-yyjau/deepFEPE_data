""" For use in dumping single frame ground truths of KITTI Odometry Dataset
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
import cv2
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
    load_velo_scan,
    rectify,
    read_calib_file,
    transform_from_rot_trans,
    scale_intrinsics,
    scale_P,
)
from utils_kitti import load_as_float, load_as_array, load_sift, load_SP

import dsac_tools.utils_misc as utils_misc
from dsac_tools.utils_misc import crop_or_pad_choice

from glob import glob
import yaml

DEEPSFM_PATH = "/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm"
sys.path.append(DEEPSFM_PATH)


class kitti_seq_loader(object):
    def __init__(
        self,
        dataset_dir,
        img_height=375,
        img_width=1242,
        cam_ids=["02"],
        get_X=False,
        get_pose=False,
        get_sift=False,
        get_SP=False,
        sift_num=2000,
        if_BF_matcher=False,
        save_npy=True,
        delta_ijs=[1]
    ):
        # depth_size_ratio=1):
        # dir_path = Path(__file__).realpath().dirname()

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = cam_ids
        # assert self.cam_ids == ['02'], 'Support left camera only!'
        self.cid_to_num = {"00": 0, "01": 1, "02": 2, "03": 3}
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]
        # self.train_seqs = [4]
        # self.test_seqs = []
        # self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # self.test_seqs = []
        self.map_to_raw = { # for calibration files
            "00": "2011_10_03_drive_0027",
            "01": "2011_10_03_drive_0042",
            "02": "2011_10_03_drive_0034",
            "03": "2011_09_26_drive_0067",
            "04": "2011_09_30_drive_0016",
            "05": "2011_09_30_drive_0018",
            "06": "2011_09_30_drive_0020",
            "07": "2011_09_30_drive_0027",
            "08": "2011_09_30_drive_0028",
            "09": "2011_09_30_drive_0033",
            "10": "2011_09_30_drive_0034",
        }

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
            # self.bf = cv2.BFMatcher(normType=cv2.NORM_L2)
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            # search_params = dict(checks = 50)
            # self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            # self.sift_matcher = self.bf if BF_matcher else self.flann

        self.scenes = {"train": [], "test": []}
        if self.get_SP:
            self.prapare_SP()
        self.collect_train_folders()
        self.collect_test_folders()
        self.train_scenes = {}

    # deprecated
    def prapare_SP(self):
        logging.info("Preparing SP inference.")
        with open(DEEPSFM_PATH + "/configs/superpoint_coco_train.yaml", "r") as f:
            self.config_SP = yaml.load(f, Loader=yaml.FullLoader)
            nms_dist = self.config_SP["model"]["nms"]
            conf_thresh = self.config_SP["model"]["detection_threshold"]
            # nn_thresh = config_SP['model']['nn_thresh']
            nn_thresh = 1.0
            path = DEEPSFM_PATH + "/" + self.config_SP["pretrained"]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.fe = SuperPointFrontend_torch(
                weights_path=path,
                nms_dist=nms_dist,
                conf_thresh=conf_thresh,
                nn_thresh=nn_thresh,
                cuda=False,
                device=device,
            )

    def collect_train_folders(self):
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, "sequences", "%.2d" % seq)
            self.scenes["train"].append(seq_dir)

    def collect_test_folders(self):
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, "sequences", "%.2d" % seq)
            self.scenes["test"].append(seq_dir)

    def read_images_files_from_folder(self, drive_path, scene_data):
        img_dir = os.path.join(drive_path, "image_%d" % scene_data["cid_num"])
        img_files = sorted(glob(img_dir + "/*.png"))
        return img_files

    def collect_scene_from_drive(self, drive_path, split="train"):
        train_scenes = []
        logging.info("Gathering info for %s..." % drive_path)
        for c in self.cam_ids:
            scene_data = {
                "cid": c,
                "cid_num": self.cid_to_num[c],
                "dir": Path(drive_path),
                "rel_path": Path(drive_path).name + "_" + c,
                # "rel_path": str(Path(drive_path).name + "_" + c)[-5:],
            }
            # print(f"scene_data: {scene_data}")
            # img_dir = os.path.join(drive_path, 'image_%d'%scene_data['cid_num'])
            # scene_data['img_files'] = sorted(glob(img_dir + '/*.png'))
            scene_data["img_files"] = self.read_images_files_from_folder(
                drive_path, scene_data
            )
            scene_data["N_frames"] = len(scene_data["img_files"])
            assert scene_data["N_frames"] != 0, "No file found for %s!" % drive_path
            scene_data["frame_ids"] = [
                "{:06d}".format(i) for i in range(scene_data["N_frames"])
            ]

            img_shape = None
            zoom_xy = None
            show_zoom_info = True
            for idx in tqdm(range(scene_data["N_frames"])):
                img, zoom_xy, _ = self.load_image(scene_data, idx, show_zoom_info)
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
            P_rect_ori_dict = self.get_P_rect(scene_data, scene_data["calibs"])
            intrinsics = P_rect_ori_dict[c][:, :3]
            calibs_rects = self.get_rect_cams(intrinsics, P_rect_ori_dict["02"])

            drive_in_raw = self.map_to_raw[drive_path[-2:]]
            date = drive_in_raw[:10]
            seq = drive_in_raw[-4:]
            calib_path_in_raw = Path(self.dataset_dir) / "raw" / date
            imu2velo_dict = read_calib_file(calib_path_in_raw / "calib_imu_to_velo.txt")
            velo2cam_dict = read_calib_file(calib_path_in_raw / "calib_velo_to_cam.txt")
            cam2cam_dict = read_calib_file(calib_path_in_raw / "calib_cam_to_cam.txt")
            velo2cam_mat = transform_from_rot_trans(
                velo2cam_dict["R"], velo2cam_dict["T"]
            )
            imu2velo_mat = transform_from_rot_trans(
                imu2velo_dict["R"], imu2velo_dict["T"]
            )
            cam_2rect_mat = transform_from_rot_trans(
                cam2cam_dict["R_rect_00"], np.zeros(3)
            )
            scene_data["calibs"].update(
                {
                    "K": intrinsics,
                    "P_rect_ori_dict": P_rect_ori_dict,
                    "cam_2rect": cam_2rect_mat,
                    "velo2cam": velo2cam_mat,
                }
            )
            scene_data["calibs"].update(calibs_rects)

            # Get pose
            poses = (
                np.genfromtxt(
                    self.dataset_dir / "poses" / "{}.txt".format(drive_path[-2:])
                )
                .astype(np.float32)
                .reshape(-1, 3, 4)
            )
            assert scene_data["N_frames"] == poses.shape[0], (
                "scene_data[N_frames]!=poses.shape[0], %d!=%d"
                % (scene_data["N_frames"], poses.shape[0])
            )
            scene_data["poses"] = poses

            # cam to gt?
            scene_data["Rt_cam2_gt"] = scene_data["calibs"]["Rtl_gt"]

            train_scenes.append(scene_data)
        self.train_scenes = train_scenes
        return train_scenes

    def construct_sample(self, scene_data, idx, frame_id, show_zoom_info):
        img, zoom_xy, img_ori = self.load_image(scene_data, idx, show_zoom_info)
        # print(img.shape, img_ori.shape)
        sample = {"img": img, "id": frame_id}

        # get 3d points
        if self.get_X:
            # feed in intrinsics for TUM to extract depth
            velo = self.load_velo(scene_data, idx, scene_data["calibs"].get("P_rect_noScale", None))
            # print(f"velo: {velo.shape}")
            if velo is None:
                logging.error("0 velo in %s. Skipped." % scene_data["dir"])
            # change to homography
            velo_homo = utils_misc.homo_np(velo)
            logging.debug(f"velo_homo: {velo_homo.shape}")
            val_idxes, X_rect, X_cam0 = rectify(
                velo_homo, scene_data["calibs"]
            )  # list, [N, 3]
            logging.debug(f"X_rect: {X_rect.shape}")
            logging.debug(f"X_cam0: {X_cam0.shape}")
            logging.debug(f"val_idxes: {len(val_idxes)}")

            sample["X_cam2_vis"] = X_rect[val_idxes].astype(np.float32)
            sample["X_cam0_vis"] = X_cam0[val_idxes].astype(np.float32)
        if self.get_pose:
            sample["pose"] = scene_data["poses"][idx].astype(np.float32)
        if self.get_sift:
            # logging.info('Getting sift for frame %d/%d.'%(idx, scene_data['N_frames']))
            kp, des = self.sift.detectAndCompute(
                img_ori, None
            )  ## IMPORTANT: normalize these points
            x_all = np.array([p.pt for p in kp])
            # print(zoom_xy)
            x_all = (x_all * np.array([[zoom_xy[0], zoom_xy[1]]])).astype(np.float32)
            # print(x_all.shape, np.amax(x_all, axis=0), np.amin(x_all, axis=0))
            if x_all.shape[0] != self.sift_num:
                choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)
                x_all = x_all[choice]
                des = des[choice]
            sample["sift_kp"] = x_all
            sample["sift_des"] = des
        if self.get_SP:
            img_ori_gray = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)
            img = (
                torch.from_numpy(img_ori_gray).float().unsqueeze(0).unsqueeze(0).float()
                / 255.0
            )
            pts, desc, _, heatmap = self.fe.run(img)
            pts = pts[0].T  # [N, 3]
            pts[:, :2] = (pts[:, :2] * np.array([[zoom_xy[0], zoom_xy[1]]])).astype(
                np.float32
            )
            desc = desc[0].T  # [N, 256]
            sample["SP_kp"] = pts
            sample["SP_des"] = desc
        return sample

    def dump_drive(self, args, drive_path, split, scene_data=None):
        """ main entry point, dump dataset from drive_path

        """
        assert split in ["train", "test"]
        # get scene data
        if scene_data is None:
            train_scenes = self.collect_scene_from_drive(drive_path, split=split)
            if not train_scenes:
                logging.warning("Empty scene data for %s. Skipped." % drive_path)
                return
            assert (
                len(train_scenes) == 1
            ), "More than one camera not supported! %d" % len(train_scenes)
            scene_data = train_scenes[0]

        # create dump folders
        # dump_dir = Path(args.dump_root) / scene_data["rel_path"][-5:]
        dump_dir = Path(args.dump_root) / scene_data["rel_path"]
        print(f"dump_dir: {dump_dir}")
        if dump_dir.is_dir():
            logging.warning(f'dump_root exists: {dump_dir}')
        else:
            dump_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f'dump_root created: {dump_dir}')
        intrinsics = scene_data["calibs"]["K"]
        # dump_cam_file = dump_dir / "cam"
        # save: intrinsics
        np.save(dump_dir /"cam.npy", intrinsics.astype(np.float32))
        dump_Rt_cam2_gt_file = dump_dir / "Rt_cam2_gt"
        # save: camera intrinsics * extrinsics
        np.save(dump_Rt_cam2_gt_file, scene_data["Rt_cam2_gt"].astype(np.float32))
        poses_file = dump_dir / "poses"
        poses = []

        logging.info("Dumping %d samples to %s..." % (scene_data["N_frames"], dump_dir))
        sample_name_list = []
        # sift_des_list = []
        # dump features, images frame by frame
        for idx in tqdm(range(scene_data["N_frames"])):
            frame_id = scene_data["frame_ids"][idx]
            assert int(frame_id) == idx
            sample = self.construct_sample(
                scene_data, idx, frame_id, show_zoom_info=False
            )

            img, frame_nb = sample["img"], sample["id"]
            dump_img_file = dump_dir / "{}.jpg".format(frame_nb)
            # scipy.misc.imsave(dump_img_file, img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dump_img_file), img)
            
            if "pose" in sample.keys():
                poses.append(sample["pose"].astype(np.float32))
            if "X_cam0_vis" in sample.keys():
                dump_X_cam0_file = dump_dir / "X_cam0_{}".format(frame_nb)
                dump_X_cam2_file = dump_dir / "X_cam2_{}".format(frame_nb)
                if self.save_npy:
                    np.save(str(dump_X_cam0_file) + ".npy", sample["X_cam0_vis"])
                    np.save(str(dump_X_cam2_file) + ".npy", sample["X_cam2_vis"])
                else:
                    saveh5(
                        {
                            "X_cam0_vis": sample["X_cam0_vis"],
                            "X_cam2_vis": sample["X_cam2_vis"],
                        },
                        f'{dump_X_file}/.h5',
                    )
            if "sift_kp" in sample.keys():
                dump_sift_file = dump_dir / "sift_{}".format(frame_nb)
                if self.save_npy:
                    np.save(
                        f"{str(dump_sift_file)}.npy",
                        np.hstack((sample["sift_kp"], sample["sift_des"])),
                    )
                else:
                    saveh5(
                        {"sift_kp": sample["sift_kp"], "sift_des": sample["sift_des"]},
                        f"{str(dump_sift_file)}.h5",
                    )
                # sift_des_list.append(sample['sift_des'])
            if "SP_kp" in sample.keys():
                dump_sift_file = dump_dir / "SP_{}".format(frame_nb)
                if self.save_npy:
                    np.save(
                        f"{str(dump_sift_file)}.npy",
                        np.hstack((sample["SP_kp"], sample["SP_des"])),
                    )
                    # print(sample['SP_kp'].shape, sample['SP_des'].shape)
                else:
                    pass

            # sample_name_list.append("%s %s" % (str(dump_dir)[-5:], frame_nb))
            sample_name_list.append(f"{str(scene_data['rel_path'])} {frame_nb}")
            logging.debug(f"sample_name_list: {sample_name_list[-1]}")
        # Get all poses
        if "pose" in sample.keys():
            if len(poses) != 0:
                # np.savetxt(poses_file, np.array(poses).reshape(-1, 16), fmt='%.20e')a
                if self.save_npy:
                    np.save(str(poses_file) + ".npy", np.stack(poses).reshape(-1, 3, 4))
                else:
                    saveh5(
                        {"poses": np.array(poses).reshape(-1, 3, 4)}, str(poses_file) + ".h5"
                    )

        # Get SIFT matches
        delta_ijs = self.delta_ijs
        if self.get_sift:
            # delta_ijs = [1, 2, 3, 5, 8, 10]
            # delta_ijs = [1]
            num_tasks = len(delta_ijs)
            num_workers = min(len(delta_ijs), default_number_of_process)
            # num_workers = 1
            logging.info(
                "Getting SIFT matches on %d workers for delta_ijs = %s"
                % (num_workers, " ".join(str(e) for e in delta_ijs))
            )

            with ProcessPool(max_workers=num_workers) as pool:
                tasks = pool.map(
                    dump_sift_match_idx,
                    delta_ijs,
                    [scene_data["N_frames"]] * num_tasks,
                    [dump_dir] * num_tasks,
                    [self.save_npy] * num_tasks,
                    [self.if_BF_matcher] * num_tasks,
                )
                try:
                    for _ in tqdm(tasks.result(), total=num_tasks):
                        pass
                except KeyboardInterrupt as e:
                    tasks.cancel()
                    raise e

        # Get SP matches
        if self.get_SP:
            delta_ijs = [1, 2, 3, 5, 8, 10]
            nn_threshes = [0.7, 1.0]
            # delta_ijs = [1]
            num_tasks = len(delta_ijs)
            num_workers = min(len(delta_ijs), default_number_of_process)
            # num_workers = 1
            logging.info(
                "Getting SP matches on %d workers for delta_ijs = %s"
                % (num_workers, " ".join(str(e) for e in delta_ijs))
            )

            with ProcessPool(max_workers=num_workers) as pool:
                tasks = pool.map(
                    dump_SP_match_idx,
                    delta_ijs,
                    [scene_data["N_frames"]] * num_tasks,
                    [dump_dir] * num_tasks,
                    [self.save_npy] * num_tasks,
                    [nn_threshes] * num_tasks,
                )
                try:
                    for _ in tqdm(tasks.result(), total=num_tasks):
                        pass
                except KeyboardInterrupt as e:
                    tasks.cancel()
                    raise e

            # for delta_ij in delta_ijs:
            #     dump_match_idx(delta_ij, scene_data['N_frames'], sift_des_list, dump_dir, self.save_npy, self.if_BF_matcher)

        if len(list(dump_dir.rglob("*.jpg"))) < 2:
            # dump_dir.rmtree()
            import shutil
            shutil.rmtree(dump_dir)

        return sample_name_list

    def load_image(self, scene_data, tgt_idx, show_zoom_info=True):
        img_file = (
            scene_data["dir"]
            / "image_{}".format(scene_data["cid_num"])
            / f'{str(scene_data["frame_ids"][tgt_idx])}.png'
        )
        if not img_file.is_file():
            logging.warning("Image %s not found!" % img_file)
            return None, None, None
        # img_ori = scipy.misc.imread(img_file)
        # print(f"img_file: {img_file}")
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
            img = cv2.resize(img_ori, (self.img_height, self.img_width))
            return img, (zoom_x, zoom_y), img_ori

    def get_P_rect(self, scene_data, calibs, get_2cam_dict=True):
        # calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'
        calib_file = scene_data["dir"] / "calib.txt"
        if get_2cam_dict:
            P_rect = {}
            for cid in ["00", "01", "02", "03"]:
                P_rect[cid], _ = read_odo_calib_file(
                    calib_file, cid=self.cid_to_num[cid]
                )
                if calibs["rescale"]:
                    P_rect[cid] = scale_P(
                        P_rect[cid], calibs["zoom_xy"][0], calibs["zoom_xy"][1]
                    )
            return P_rect
        else:
            P_rect, _ = read_odo_calib_file(calib_file, cid=self.cid_to_num[cid])
            if calibs["rescale"]:
                P_rect = scale_P(P_rect, calibs["zoom_xy"][0], calibs["zoom_xy"][1])
        return P_rect

    def get_rect_cams(self, K, P_rect_20):
        Ml_gt = np.matmul(np.linalg.inv(K), P_rect_20)
        tl_gt = Ml_gt[:, 3:4]
        Rl_gt = Ml_gt[:, :3]
        Rtl_gt = np.vstack(
            (
                np.hstack((Rl_gt, tl_gt)),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
        )
        calibs_rects = {"Rtl_gt": Rtl_gt}
        return calibs_rects

    @staticmethod
    def load_velo(scene_data, tgt_idx, calib_K=None):
        velo_file = (
            scene_data["dir"] / "velodyne" / f'{scene_data["frame_ids"][tgt_idx]}.bin'
        )
        if not velo_file.is_file():
            logging.warning("Velo file %s not found!" % velo_file)
            return None
        velo = load_velo_scan(str(velo_file))[:, :3]
        return velo


def dump_sift_match_idx(delta_ij, N_frames, dump_dir, save_npy, if_BF_matcher):
    # select which kind of matcher
    if (
        if_BF_matcher
    ):  # OpenCV sift matcher must be created inside each thread (because it does not support sharing across threads!)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2)
        sift_matcher = bf
    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        sift_matcher = flann

    # do matching for different deltas
    for ii in tqdm(range(N_frames - delta_ij)):
        jj = ii + delta_ij

        ## read sift features and descriptors for 2 frames
        sift_kps_ii, sift_des_ii = load_sift(
            dump_dir, "%06d" % ii, ext=".npy" if save_npy else ".h5"
        )
        sift_kps_jj, sift_des_jj = load_sift(
            dump_dir, "%06d" % jj, ext=".npy" if save_npy else ".h5"
        )

        # all_ij, good_ij = get_sift_match_idx_pair(sift_matcher, sift_des_list[ii], sift_des_list[jj])
        all_ij, good_ij, quality_good, quality_all = get_sift_match_idx_pair(
            sift_matcher, sift_des_ii.copy(), sift_des_jj.copy()
        )
        if all_ij is None:
            logging.warning(
                "KNN match failed dumping %s frame %d-%d. Skipping" % (dump_dir, ii, jj)
            )
            continue
        dump_ij_idx_file = dump_dir / "ij_idx_{}-{}".format(ii, jj)
        dump_ij_quality_file = dump_dir / "ij_quality_{}-{}".format(ii, jj)
        dump_ij_match_quality_file = dump_dir / "ij_match_quality_{}-{}".format(ii, jj)

        if save_npy:
            np.save(str(dump_ij_idx_file) + "_all.npy", all_ij)
            np.save(str(dump_ij_idx_file) + "_good.npy", good_ij)
            np.save(str(dump_ij_quality_file) + "_good.npy", quality_good)
            np.save(str(dump_ij_quality_file) + "_all.npy", quality_all)

            # print(good_ij, good_ij.shape)
            match_quality_good = np.hstack(
                (sift_kps_ii[good_ij[:, 0]], sift_kps_jj[good_ij[:, 1]], quality_good)
            )  # [[x1, y1, x2, y2, dist_good, ratio_good]]
            match_quality_all = np.hstack(
                (sift_kps_ii[all_ij[:, 0]], sift_kps_jj[all_ij[:, 1]], quality_all)
            )  # [[x1, y1, x2, y2, dist_good, ratio_good]]
            np.save(str(dump_ij_match_quality_file) + "_good.npy", match_quality_good)
            np.save(str(dump_ij_match_quality_file) + "_all.npy", match_quality_all)

            # print(good_ij.dtype, quality_good.dtype, good_ij.shape, quality_good.shape)
        else:
            dump_ij_idx_dict = {"all_ij": all_ij, "good_ij": good_ij}
            saveh5(dump_ij_idx_dict, str(dump_ij_idx_file) + ".h5")


def get_sift_match_idx_pair(sift_matcher, des1, des2):
    """
    do matchings, test the quality of matchings
    """
    try:
        matches = sift_matcher.knnMatch(
            des1, des2, k=2
        )  # another option is https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py#L309
    except Exception as e:
        logging.error(traceback.format_exception(*sys.exc_info()))
        return None, None
    # store all the good matches as per Lowe's ratio test.
    good = []
    all_m = []
    quality_good = []
    quality_all = []
    for m, n in matches:
        all_m.append(m)
        if m.distance < 0.8 * n.distance:
            good.append(m)
            quality_good.append([m.distance, m.distance / n.distance])
        quality_all.append([m.distance, m.distance / n.distance])

    good_ij = [[mat.queryIdx for mat in good], [mat.trainIdx for mat in good]]
    all_ij = [[mat.queryIdx for mat in all_m], [mat.trainIdx for mat in all_m]]
    return (
        np.asarray(all_ij, dtype=np.int32).T.copy(),
        np.asarray(good_ij, dtype=np.int32).T.copy(),
        np.asarray(quality_good, dtype=np.float32).copy(),
        np.asarray(quality_all, dtype=np.float32).copy(),
    )


def dump_SP_match_idx(delta_ij, N_frames, dump_dir, save_npy, nn_threshes):
    for nn_thresh, name in zip(nn_threshes, ["good", "all"]):
        SP_matcher = PointTracker(max_length=2, nn_thresh=nn_thresh)

        for ii in tqdm(range(N_frames - delta_ij)):
            jj = ii + delta_ij

            SP_kps_ii, SP_des_ii = load_SP(
                dump_dir, "%06d" % ii, ext=".npy" if save_npy else ".h5"
            )
            SP_kps_jj, SP_des_jj = load_SP(
                dump_dir, "%06d" % jj, ext=".npy" if save_npy else ".h5"
            )

            matches, scores = get_SP_match_idx_pair(
                SP_matcher, SP_kps_ii, SP_kps_jj, SP_des_ii, SP_des_jj
            )

            dump_ij_match_quality_file = dump_dir / "SP_ij_match_quality_{}-{}".format(
                ii, jj
            )

            if save_npy:
                # print(matches.shape, scores.shape)
                match_quality = np.hstack(
                    (matches, scores)
                )  # [[x1, y1, x2, y2, dist_good, ratio_good]]
                np.save(str(dump_ij_match_quality_file) + "_%s.npy" % name, match_quality)
            else:
                pass


def get_SP_match_idx_pair(matcher, kps1, kps2, des1, des2):
    matcher.update(kps1.T, des1.T)
    matcher.update(kps2.T, des2.T)
    matches = matcher.get_matches().T  # [N, 4]

    scores = matcher.mscores[-1, :].reshape(-1, 1)  # [N, 1]

    return matches.astype(np.float32).copy(), scores.astype(np.float32).copy()




def read_odo_calib_file(filepath, cid=2):
    # From https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_odom_loader.py#L133
    """Read in a calibration file and parse into a dictionary."""
    with open(filepath, "r") as f:
        C = f.readlines()

    def parseLine(L, shape):
        data = L.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data

    proj_c2p = parseLine(C[cid], shape=(3, 4))
    proj_v2c = parseLine(C[-1], shape=(3, 4))
    filler = np.array([0, 0, 0, 1]).reshape((1, 4))
    proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
    return proj_c2p, proj_v2c
