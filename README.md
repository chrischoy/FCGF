# Fully Convolutional Geometric Features, ICCV, 2019

Extracting geometric features from 3D scans or point clouds is the first step in applications such as registration, reconstruction, and tracking. State-of-the-art methods require computing low-level features as input or extracting patch-based features with limited receptive field. In this work, we present fully-convolutional geometric features, computed in a single pass by a 3D fully-convolutional network. We also present new metric learning losses that dramatically improve performance. Fully-convolutional geometric features are compact, capture broad spatial context, and scale to large scenes. We experimentally validate our approach on both indoor and outdoor datasets. Fully-convolutional geometric features achieve state-of-the-art accuracy without requiring prepossessing, are compact (32 dimensions), and are 600 times faster than the most accurate prior method.

[ICCV'19 Paper](https://node1.chrischoy.org/data/publications/fcgf/fcgf.pdf)

## News

- 2020-10-02 Measure the FCGF speedup on v0.5 on [MinkowskiEngineBenchmark](https://github.com/chrischoy/MinkowskiEngineBenchmark). The speedup ranges from 2.7x to 7.7x depending on the batch size.
- 2020-09-04 Updates on ME v0.5 further speed up the inference time from 13.2ms to 11.8ms. As a reference, ME v0.4 takes 37ms.
- 2020-08-18 Merged the v0.5 to the master with v0.5 installation. You can now use the full GPU support for sparse tensor hi-COO representation for faster training and inference.
- 2020-08-07 [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.5 improves the **FCGF inference speed by x2.8** (280% speed-up, feed forward time for ResUNetBN2C on the 3DMatch kitchen point cloud ID-20: 37ms (ME v0.4.3) down to 13.2ms (ME v0.5.0). Measured on TitanXP, Ryzen-3700X).
- 2020-06-15 [Source code](https://github.com/chrischoy/DeepGlobalRegistration) for **Deep Global Registration, CVPR'20 Oral** has been released. Please refer to the repository and the paper for using FCGF for registration.

## 3D Feature Accuracy vs. Speed

|   Comparison Table           | Speed vs. Accuracy |
|:----------------------------:|:------------------:|
| ![Table](assets/table.png)   | ![Accuracy vs. Speed](assets/fps_acc.png) |

*Feature-match recall and speed in log scale on the 3DMatch benchmark. Our approach is the most accurate and the fastest. The gray region shows the Pareto frontier of the prior methods.*


### Related Works

3DMatch by Zeng et al. uses a siamese convolutional network to learn 3D patch descriptors.
CGF by Khoury et al. maps 3D oriented histograms to a low-dimensional feature space using multi-layer perceptrons. PPFNet and PPF FoldNet by Deng et al. adapts the PointNet architecture for geometric feature description. 3DFeat by Yew and Lee uses a PointNet to extract features in outdoor scenes.

Our work addressed a number of limitations in the prior work. First, all prior approaches extract a small 3D patch or a set of points and map it to a low-dimensional space. This not only limits the receptive field of the network but is also computationally inefficient since all intermediate representations are computed separately even for overlapping 3D regions. Second, using expensive low-level geometric signatures as input can slow down feature computation. Lastly, limiting feature extraction to a subset of interest points results in lower spatial resolution for subsequent matching stages and can thus reduce registration accuracy.


### Fully Convolutional Metric Learning  and Hardest Contrastive, Hardest Triplet Loss

Traditional metric learning assumes that the features are independent and identically distributed (i.i.d.) since a batch is constructed by random sampling. However, in fully-convolutional feature extraction first proposed in Universal Correspondence Network, Choy 2016, adjacent features are locally correlated and hard-negative mining could find features adjacent to anchors, which are false negatives. Thus, filtering out these false negatives is a crucial step similar to how Universal Correspondence Network by Choy et al. used a distance threshold to filter out the false negatives.

Also, the number of features used in the fully-convolutional setting is orders of magnitude larger than that in standard metric learning algorithms. For instance, FCGF generates ~40k features for a pair of scans (this increases proportionally with the batch size) while a minibatch in traditional metric learning has around 1k features. Thus, it is not feasible to use all pairwise distances within a batch in the standard metric learning.

Instead, we propose the hardest-contrastive loss and the hardest-triplet loss. Visually, these are simple variants that use the hardest negatives for both features within a positive pair.
One of the key advantages of the hardest-contrastive loss is that you do not need to save the temporary variables used to find the hardest negatives. This small change allows us to reconstruct the loss from the hardest negatives indices and throw away the intermediate results among a large number of feature. [Here](https://github.com/chrischoy/open-ucn/blob/master/lib/ucn_trainer.py#L435), we used almost 40k features to mine the hardest negative and destroy all intermediate variables once the indices of the hardest negatives are found for each positive feature.

| Contrastive Loss   | Triplet Loss       | Hardest Contrastive | Hardest Triplet    |
|:------------------:|:------------------:|:-------------------:|:------------------:|
| ![1](assets/1.png) | ![2](assets/2.png) | ![3](assets/3.png)  | ![4](assets/4.png) |

*Sampling and negative-mining strategy for each method. Blue: positives, Red: Negatives. Traditional contrastive and triplet losses use random sampling. Our hardest-contrastive and hardest-triplet losses use the hardest negatives.*

Please refer to our [ICCV'19 paper](https://node1.chrischoy.org/data/publications/fcgf/fcgf.pdf) for more details.


### Visualization of FCGF

We color-coded FCGF features for pairs of 3D scans that are 10m apart for KITTI and a 3DMatch benchmark pair for indoor scans. FCGF features are mapped to a scalar space using t-SNE and colorized with the Spectral color map.

| KITTI LIDAR Scan 1   | KITTI LIDAR Scan 2   |
|:--------------------:|:--------------------:|
| ![0](assets/3_1.png) | ![1](assets/3_2.png) |

| Indoor Scan 1              | Indoor Scan 2              |
|:--------------------------:|:--------------------------:|
| ![0](assets/kitchen_0.png) | ![1](assets/kitchen_1.png) |

#### FCGF Correspondence Visualizations

Please follow the link [Youtube Video](https://www.youtube.com/watch?v=d0p0eTaB50k) or click the image to view the YouTube video of FCGF visualizations.
[![](assets/text_scene000.gif)](https://www.youtube.com/watch?v=d0p0eTaB50k)

## Requirements

- Ubuntu 14.04 or higher
- CUDA 10.2 or higher
- Python v3.7 or higher
- Pytorch v1.5 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 or higher


## Installation & Dataset Download


We recommend conda for installation. First, create a conda environment with pytorch 1.5 or higher with

```
conda create -n py3-fcgf python=3.7
conda activate py3-fcgf
conda install pytorch -c pytorch
pip install git+https://github.com/NVIDIA/MinkowskiEngine.git
```

Next, download FCGF git repository and install the requirement from the FCGF root directory..

```
git clone https://github.com/chrischoy/FCGF.git
cd FCGF
# Do the following inside the conda environment
pip install -r requirements.txt
```

For training, download the preprocessed 3DMatch benchmark dataset.

```
./scripts/download_datasets.sh /path/to/dataset/download/dir
```

For KITTI training, follow the instruction on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the KITTI odometry training set.


## Demo: Extracting and color coding FCGF

After installation, you can run the demo script by

```
python demo.py
```

The demo script will first extract FCGF features from a mesh file generated from a kitchen scene. Next, it will color code the features independent of their spatial location.
After the color mapping using TSNE, the demo script will visualize the color coded features by coloring the input point cloud.

![demo](./assets/demo.png)

*You may have to rotate the scene to get the above visualization.*


## Training and running 3DMatch benchmark

```
python train.py --threed_match_dir /path/to/threedmatch/
```

For benchmarking the trained weights on 3DMatch, download the 3DMatch Geometric Registration Benchmark dataset from [here](http://3dmatch.cs.princeton.edu/) or run

```
bash ./scripts/download_3dmatch_test.sh /path/to/threedmatch_test/
```

and follow:

```
python -m scripts.benchmark_3dmatch.py \
    --source /path/to/threedmatch \
    --target ./features_tmp/ \
    --voxel_size 0.025 \
    --model ~/outputs/checkpoint.pth \
    --extract_features --evaluate_feature_match_recall --with_cuda
```


## Training and testing on KITTI Odometry custom split

For KITTI training, follow the instruction on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the KITTI odometry training set.

```
export KITTI_PATH=/path/to/kitti/; ./scripts/train_fcgf_kitti.sh
```

## Registration Test on 3DMatch



## Model Zoo

| Model       | Normalized Feature  | Dataset | Voxel Size    | Feature Dimension | Performance                | Link   |
|:-----------:|:-------------------:|:-------:|:-------------:|:-----------------:|:--------------------------:|:------:|
| ResUNetBN2C | True                | 3DMatch | 2.5cm (0.025) | 32                | FMR: 0.9578 +- 0.0272      | [download](https://node1.chrischoy.org/data/publications/fcgf/2019-08-19_06-17-41.pth) |
| ResUNetBN2C | True                | 3DMatch | 2.5cm (0.025) | 16                | FMR: 0.9442 +- 0.0345      | [download](https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth) |
| ResUNetBN2C | True                | 3DMatch | 5cm   (0.05)  | 32                | FMR: 0.9372 +- 0.0332      | [download](https://node1.chrischoy.org/data/publications/fcgf/2019-08-16_19-21-47.pth) |
| ResUNetBN2C | False               | KITTI   | 20cm  (0.2)   | 32                | RTE: 0.0534m, RRE: 0.1704° | [download](https://node1.chrischoy.org/data/publications/fcgf/2019-07-31_19-30-19.pth) |
| ResUNetBN2C | False               | KITTI   | 30cm  (0.3)   | 32                | RTE: 0.0607m, RRE: 0.2280° | [download](https://node1.chrischoy.org/data/publications/fcgf/2019-07-31_19-37-00.pth) |
| ResUNetBN2C | True                | KITTI   | 30cm  (0.3)   | 16                | RTE: 0.0670m, RRE: 0.2295° | [download](https://node1.chrischoy.org/data/publications/fcgf/KITTI-v0.3-ResUNetBN2C-conv1-5-nout16.pth) |
| ResUNetBN2C | True                | KITTI   | 30cm  (0.3)   | 32                | RTE: 0.0639m, RRE: 0.2253° | [download](https://node1.chrischoy.org/data/publications/fcgf/KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth) |


## Raw Data for FCGF Figure 4

- [Distance threshold data](https://raw.githubusercontent.com/chrischoy/FCGF/master/assets/fig4_dist_thresh.txt)
- [Inlier threshold data](https://raw.githubusercontent.com/chrischoy/FCGF/master/assets/fig4_inlier_thresh.txt)


## Citing FCGF

FCGF will be presented at ICCV'19: Friday, November 1, 2019, 1030–1300 Poster 4.1 (Hall B)

```
@inproceedings{FCGF2019,
    author = {Christopher Choy and Jaesik Park and Vladlen Koltun},
    title = {Fully Convolutional Geometric Features},
    booktitle = {ICCV},
    year = {2019},
}
```

## Related Projects

- A neural network library for high-dimensional sparse tensors: [Minkowski Engine, CVPR'19](https://github.com/StanfordVL/MinkowskiEngine)
- Semantic segmentation on a high-dimensional sparse tensor: [4D Spatio Temporal ConvNets, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- The first fully convolutional metric learning for correspondences: [Universal Correspondence Network, NIPS'16](https://github.com/chrischoy/open-ucn)
- 3D Registration Network with 6-dimensional ConvNets: [Deep Global Registration, CVPR'20](https://github.com/chrischoy/DeepGlobalRegistration)


## Projects using FCGF

- Gojcic et al., [Learning multiview 3D point cloud registration, CVPR'20](https://arxiv.org/abs/2001.05119)
- Choy et al., [Deep Global Registration, CVPR'20 Oral](https://arxiv.org/abs/2004.11540)


## Acknowledgements

We want to thank all the ICCV reviewers, especially R2, for suggestions and valuable pointers.
