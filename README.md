# Fully Convolutional Geometric Features, ICCV, 2019

Extracting geometric features from 3D scans or point clouds is the first step in applications such as registration, reconstruction, and tracking. State-of-the-art methods require computing low-level features as input or extracting patch-based features with limited receptive field. In this work, we present fully-convolutional geometric features, computed in a single pass by a 3D fully-convolutional network. We also present new metric learning losses that dramatically improve performance. Fully-convolutional geometric features are compact, capture broad spatial context, and scale to large scenes. We experimentally validate our approach on both indoor and outdoor datasets. Fully-convolutional geometric features achieve state-of-the-art accuracy without requiring prepossessing, are compact (32 dimensions), and are 600 times faster than the most accurate prior method.


## 3D Feature Accuracy vs. Speed

![Accuracy vs. Speed](images/fps_acc.png)
*Feature-match recall and speed in log scale on the 3DMatch benchmark. Our approach is the most accurate and the fastest. The gray region shows the Pareto frontier of the prior methods.*


## Installation & Dataset Download


```
./scripts/download_datasets.sh /path/to/dataset/download/dir
```

Follow the instruction on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the KITTI odometry training set.


## Training

```
python train.py --threed_match_dir /path/to/threedmatch/
```

## 3DMatch Benchmark Evaluation

```
python benchmark_3dmatch.py --source /path/to/threedmatch --target ./features_tmp/ --voxel_size 0.025 --model ~/outputs/checkpoint.pth --do_generate --do_exp_feature --with_cuda
```
