# Data processor
Data processor for pose estimation or visual odometry tasks.

## KITTI dataset
### Raw data structure
- Download raw data from [here](http://www.cvlibs.net/datasets/kitti/raw_data.php).
- Download odometry data (color) from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- Copy the ground truth poses from `deepFEPE/datasets/kitti_gt_poses`.
```
`-- KITTI (raw data, odometry sequences, GT poses)
|   |-- raw
|   |   |-- 2011_09_26_drive_0020_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_09_28_drive_0001_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_09_29_drive_0004_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_09_30_drive_0016_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 2011_10_03_drive_0027_sync
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |-- sequences
|   |   |-- 00/
|   |   |-- ...
|   |   |-- 10/
|   |-- poses
|   |   |-- 00.txt
|   |   |-- ...
|   |   |-- 10.txt
`   `   `
```
### Run command
**``WE ARE NOT FILTERING STATIC FRAMES FOR THE ODO DATASET!``**
Set ``--with_pose`` ``--with_X`` ``--with_sift`` to decide whether to dump pose files, rectified lidar points, and SIFT kps/des and corres.
- Specify your `dataset_dir` and `dump_root`
```
python dump_tools/dump_data.py --dump --dataset_dir /media/yoyee/Big_re/kitti/data_odometry_color/dataset/ \
--dump_root /media/yoyee/Big_re/kitti/kitti_dump/odo_corr_dump_siftIdx_npy_delta1_test_0714 \
--with_pose --with_sift \
--img_height 376 --img_width 1241 --dataloader_name kitti_seq_loader --cam_id '02'
```

## ApolloScape dataset
### Raw data structure
- Download raw data (Training data, Road11.tar.gz) from [here](http://apolloscape.auto/self_localization.html) or use the following script.
```
python dump_tools/apollo/download.py -h
python dump_tools/apollo/download.py --dataset_dir /media/yoyee/Big_re/apollo/train_seq_1 --if_download
# change the name to Road11.tar.gz
tar zxf Road11.tar.gz
```

### Run command
- Specify your `dataset_dir` and `dump_root`
```
python dump_tools/dump_data.py  --dump --dataset_dir /media/yoyee/Big_re/apollo/train_seq_1/  --dataloader_name  apollo_train_loader  --with_pose    --with_sift --dump_root /media/yoyee/Big_re/apollo/apollo_dump/train_seq_1/  --cam_id 5  --img_height 2710 --img_width  3384 
```

## EuRoC dataset (not tested)
### Raw data
```
python dump_tools/euroc/download.py --dataset_dir /media/yoyee/Big_re/euroc/train_seq_1 --if_download --if_untar
```
### Process data 
- Match time stamps of rgb images, depth images and poses
```
python dump_tools/euroc/process_poses_euroc.py --dataset_dir /data/euroc/test2/ --dataset euroc
```
### Run command
```
python dump_tools/dump_data.py  --dump --dataset_dir  /data/euroc/raw_sequence --with_pose     --with_sift --dump_root /data/euroc/euroc_dump/slam_seq_v1   --cam_id 00 --img_height 480 --img_width 752 --dataloader_name euroc_seq_loader
```

## TUM dataset (not tested)
### Raw data
```
python dump_tools/tum/download.py 
```
### Run command
```
python dump_tools/dump_data.py  --dump --dataset_dir /data/tum/raw_sequences  --with_pose     --with_sift --dump_root /data/tum/tum_dump/slam_seq_v1 --with_X   --cam_id 00  --img_height 480 --img_width 640  --dataloader_name tum_seq_loader
```

## Visualize dataset
Refer to https://github.com/eric-yyjau/kitti_instance_RGBD_utils for some code snippets.

## Citations
Please cite the following papers.
- DeepFEPE
```
```

# Credits
This implementation is developed by [You-Yi Jau](https://github.com/eric-yyjau) and [Rui Zhu](https://github.com/Jerrypiglet). Please contact You-Yi for any problems. 

# License
DeepFEPE is released under the MIT License. 

