# extract tars

import subprocess
import glob


if __name__ == "__main__":
    # download
    sequences = [
        "freiburg1/rgbd_dataset_freiburg1_desk",
        "freiburg1/rgbd_dataset_freiburg1_desk2",
        "freiburg1/rgbd_dataset_freiburg1_room",
        "freiburg2/rgbd_dataset_freiburg2_desk",
        "freiburg2/rgbd_dataset_freiburg2_xyz",
        "freiburg3/rgbd_dataset_freiburg3_long_office_household",
        "freiburg3/rgbd_dataset_freiburg3_nostructure_texture_far",
    ]
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_far.tgz

    base_path = "https://vision.in.tum.de/rgbd/dataset/"

    if_download = True
    if_untar = True

    if if_download:
        for seq in sequences:
            subprocess.run(f"wget {base_path + seq + '.tgz'}", shell=True, check=True)

    if if_untar:
        # unzip
        tar_files = glob.glob("*.tgz")
        for f in tar_files:
            subprocess.run(f"tar -zxf {f}", shell=True, check=True)
