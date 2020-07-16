# extract tars


import subprocess
import glob
import argparse
import logging
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foo")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/data/euroc/test1/",
        help="path to dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="euroc",
        help="path to dataset",
    )
    args = parser.parse_args()
    print(args)


    if_match_stamps = True
    if_cvt_kitti = True
    if_cp_tum = False # copy tum camera models to the folder

    folders = glob.glob(f"{args.dataset_dir}/**/")
    # target = ['rgbd_dataset_freiburg2_xyz']
    # folders = [Path(args.dataset_dir) / t for t in target]
    dataset = args.dataset
    
    gt_file = "state_groundtruth_estimate0/data.csv"
    max_difference = 30
    rgb_file = 'cam0/data.csv'

    filter_ext = '_f.txt'

    for folder in folders:
        folder = folder+'mav0'
        # subprocess.run(f"tar -zxf {f}", shell=True, check=True)
        print(folder)
        ##### 
        if dataset in str(folder)[-4:]:
            print(str(folder)[-4:])
            continue
        # associate timestamps of poses

        if if_match_stamps:
            # match poses
            print("match rgb to pose")            
            subprocess.run(
                f"python associate.py {folder}/{gt_file} {folder}/{rgb_file} --max_difference {max_difference} --save_file {folder}/{rgb_file[:-4]+filter_ext}",
                shell=True,
                check=True,
            )
            # # match rgb to depth
            # print("match rgb to depth")
            # subprocess.run(
            #     f"python associate.py {folder}/depth.txt {folder}/rgb_filter.txt --max_difference 0.08 --save_file {folder}/rgb_filter.txt",
            #     shell=True,
            #     check=True,
            # )
            # # match depth to rgb
            # print("match depth to rgb")
            # subprocess.run(
            #     f"python associate.py {folder}/rgb_filter.txt {folder}/depth.txt --max_difference 0.08 --save_file {folder}/depth_filter.txt",
            #     shell=True,
            #     check=True,
            # )
            # match poses
            print("match poses to rgb")            
            subprocess.run(
                f"python associate.py {folder}/{rgb_file[:-4]+filter_ext} {folder}/{gt_file} --first_only --max_difference 0.10 --save_file {folder}/{gt_file[:-4]+filter_ext}",
                shell=True,
                check=True,
            )
            ## convert gt_file to csv format
            filename = f"{folder}/{gt_file[:-4]+filter_ext}"
            file = np.loadtxt(filename)
            np.savetxt(filename, file, fmt="%s", delimiter=",")

        if if_cvt_kitti:
            gt_file_new = gt_file[:-4]+filter_ext
            assert (
                Path(folder) / gt_file_new
            ).exists(), f"{(Path(folder) / gt_file_new)} not exists"
            # process files
            logging.info(f"generate kitti format gt pose: {folder}")
            subprocess.run(
                f"evo_traj {dataset} {str(Path(folder)/gt_file_new)} --save_as_kitti",
                shell=True,
                check=True,
            )  # https://github.com/MichaelGrupp/evo
            filename = gt_file_new[:-4] 
            filename = Path(filename).stem + ".kitti" # only get the filename
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

