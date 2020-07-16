# Deposit dataset
- **Repo:** https://github.com/Jerrypiglet/kitti_instance_RGBD_utils
- **Command:** 

## kitti
** branch: https://github.com/eric-yyjau/kitti_instance_RGBD_utils/tree/master/kitti_tools **

In *kitti_tools*:

> python dump_img_odo_tum.py --dump --dataset_dir /data/kitti/odometry --with_pose --with_X --with_sift --dump_root /newfoundland/yyjau/kitti/kitti_dump/odo_corr_dump_siftIdx_npy_delta1235810_full_1027 --num_threads=1  --img_height 376 --img_width 1241 --dataloader_name kitti_seq_loader --cam_id '02'

<!-- > python dump_img_odo.py --dump --dataset_dir /data/kitti/odometry --with_pose --with_X --with_sift --dump_root /home/ruizhu/Documents/Datasets/kitti/kitti_dump/odo_corr_dump_siftIdx_npy_delta1235810_full --num_threads=1 -->

## apollo 
** branch: https://github.com/eric-yyjau/kitti_instance_RGBD_utils/tree/dump_data/kitti_tools**


- Download dataset:
  - sequence: self Localization/ training data/ Road11.tar.gz (mainly used)
  - *Website: http://apolloscape.auto/*

- Process data 

No need to process data. Already matched.

- Dump data: (make sure to specify dimensions)

In *kitti_tools*:
> python dump_img_odo_tum.py  --dump --dataset_dir /newfoundland/yyjau/apollo/train_seq_1/  --dataloader_name  apollo_train_loader  --with_pose     --with_sift --dump_root /newfoundland/yyjau/apollo/apollo_dump/train_seq_1/   --num_threads=1  --cam_id 5  --img_height 2710 --img_width  3384

- deprecated data loader (for `self_localization_examples.tar.gz`. Images not good.)
> python dump_img_odo_tum.py  --dump --dataset_dir /newfoundland/yyjau/apollo/sample_1/  --dataloader_name  apollo_seq_loader  --with_pose     --with_sift --dump_root /newfoundland/yyjau/apollo/apollo_dump/sample_3/   --num_threads=1  --cam_id 1  --img_height 480 --img_width  600


## tum
** branch: https://github.com/eric-yyjau/kitti_instance_RGBD_utils/tree/dump_data/kitti_tools**

- Download dataset: (run the script in the folder to download)
> python kitti_tools/tum/download.py 

- Match time stamps of rgb images, depth images and poses

In *kitti_tools*:
> python process_poses.py --dataset_dir [path to dataset]

- Dump data: (make sure to specify dimensions)

In *kitti_tools*:
> python dump_img_odo_tum.py  --dump --dataset_dir /data/tum/raw_sequences  --with_pose     --with_sift --dump_root /data/tum/tum_dump/slam_seq_v1 --with_X  --num_threads=1  --cam_id 00  --img_height 480 --img_width 640  --dataloader_name tum_seq_loader

## Euroc 
** branch: https://github.com/eric-yyjau/kitti_instance_RGBD_utils/tree/dump_data/kitti_tools**

- Download dataset: (run the script in the folder to download)
> python kitti_tools/euroc/download.py 

- Process data - Match time stamps of rgb images, depth images and poses

In *kitti_tools*:
> python process_poses_euroc.py --dataset_dir /data/euroc/test2/ --dataset euroc

- Dump data: (make sure to specify dimensions)

In *kitti_tools*:
> python dump_img_odo_tum.py  --dump --dataset_dir  /data/euroc/raw_sequence --with_pose     --with_sift --dump_root /data/euroc/euroc_dump/slam_seq_v1  --num_threads=1  --cam_id 00 --img_height 480 --img_width 752 --dataloader_name euroc_seq_loader



# Run training
- **Repo:** https://github.com/eric-yyjau/deepSfm/tree/deep_F_baselineSep29
- **Command:** 

Env:
> conda create -n kitti_py36 python=3.6 pip
> conda activate kitti_py36
Install PyTorch.
> pip install -r requirements.txt
> export SHAPER_MODELS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm_ori/models/shaper/shaper/models'

Training command:
> CUDA_VISIBLE_DEVICES=1 python train_good_corr_4_vals_goodF_baseline.py train_good configs/kitti_corr_baseline.yaml temp --evalCUDA_VISIBLE_DEVICES=1 python train_good_corr_4_vals_goodF_baseline.py train_good configs/kitti_corr_baseline.yaml temp --eval

# SuperPoint input
change ``if_SP`` to ``true`` in *.config* file to enable input from SP instead of SIFT.
