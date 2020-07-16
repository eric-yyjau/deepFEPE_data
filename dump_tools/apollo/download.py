# extract tars

import subprocess
import glob
import argparse


if __name__ == "__main__":
    # download
    sequences = [
        'https://bj.bcebos.com/ad-apolloscape/self-localization/Train/Road11.tar.gz?authorization=bce-auth-v1%2F32a3c819497c4a3a948949437610ba6d%2F2019-10-23T22%3A47%3A35Z%2F604800%2F%2Fd486f24a6b5dc8439e824224cfd5559fe2e72b37e3d3dd4375c7507db18e312e'
    ]

    # base_path = "https://vision.in.tum.de/rgbd/dataset/"
    base_path = ""
    parser = argparse.ArgumentParser(description="Foo")
    parser.add_argument(
        "--dataset_dir", type=str, default="./", help="path to download dataset, need large storage"
    )
    parser.add_argument(
        "--if_download", action="store_true", default=False, help="download the dataset"
    )
    parser.add_argument(
        "--if_untar", action="store_true", default=False, help="untar the downloaded file"
    )
    args = parser.parse_args()
    print(args)

    if_download = args.if_download # True
    # if_untar = args.if_untar # True

    if if_download:
        for seq in sequences:
            command = f"wget {base_path + seq} -P {args.dataset_dir}"
            print(f"run: {command}")
            subprocess.run(command, shell=True, check=True)

    # if if_untar:
    #     # unzip
    #     tar_files = glob.glob(f"{args.dataset_dir}/*.zip")
    #     for f in tar_files:
    #         command = f"unzip {f} -d {str(f)[:-4]}"
    #         print(f"run: {command}")
    #         subprocess.run(command, shell=True, check=True)
