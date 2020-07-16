# extract tars


import subprocess
import glob
import argparse
import logging
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foo")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/data/tum/raw_sequences/",
        help="path to dataset",
    )
    args = parser.parse_args()
    print(args)

    if_match_stamps = True
    if_cvt_kitti = True
    if_cp_tum = True # copy tum camera models to the folder

    folders = glob.glob(f"{args.dataset_dir}/**/")
    target = ['rgbd_dataset_freiburg2_xyz']
    folders = [Path(args.dataset_dir) / t for t in target]

    for folder in folders:
        # subprocess.run(f"tar -zxf {f}", shell=True, check=True)
        print(folder)
        if 'tum' in str(folder)[-4:]:
            print(str(folder)[-4:])
            continue
        # associate timestamps of poses
        gt_file = "groundtruth_filter.txt"

        if if_match_stamps:
            # match poses
            print("match rgb to pose")            
            subprocess.run(
                f"python associate.py {folder}/groundtruth.txt {folder}/rgb.txt --max_difference 0.10 --save_file {folder}/rgb_filter.txt",
                shell=True,
                check=True,
            )
            # match rgb to depth
            print("match rgb to depth")
            subprocess.run(
                f"python associate.py {folder}/depth.txt {folder}/rgb_filter.txt --max_difference 0.08 --save_file {folder}/rgb_filter.txt",
                shell=True,
                check=True,
            )
            # match depth to rgb
            print("match depth to rgb")
            subprocess.run(
                f"python associate.py {folder}/rgb_filter.txt {folder}/depth.txt --max_difference 0.08 --save_file {folder}/depth_filter.txt",
                shell=True,
                check=True,
            )
            # match poses
            print("match poses to rgb")            
            subprocess.run(
                f"python associate.py {folder}/rgb_filter.txt {folder}/groundtruth.txt --first_only --max_difference 0.10 --save_file {folder}/{gt_file}",
                shell=True,
                check=True,
            )

        if if_cvt_kitti:
            assert (
                Path(folder) / gt_file
            ).exists(), f"{(Path(folder) / gt_file)} not exists"
            # process files
            logging.info(f"generate kitti format gt pose: {folder}")
            subprocess.run(
                f"evo_traj tum {str(Path(folder)/gt_file)} --save_as_kitti",
                shell=True,
                check=True,
            )  # https://github.com/MichaelGrupp/evo
            filename = gt_file[:-3] + "kitti"
            print(f"cp {filename} {Path(folder)/filename}")
            subprocess.run(
                f"cp {filename} {Path(folder)/filename}", shell=True, check=True
            )
        
    if if_cp_tum:
        # copy to the dataset folder
        print(f"copy tum/ to {args.dataset_dir}")
        subprocess.run(
            f"cp -r tum/ {args.dataset_dir}",
            shell=True,
            check=True,
        ) # https://github.com/raulmur/ORB_SLAM2

