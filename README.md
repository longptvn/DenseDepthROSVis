## DenseDepthROSVis

This is a fork of [DenseDepth](https://github.com/ialhashim/DenseDepth) made by Ibraheem Alhashim and Peter Wonka.

The purpose of this fork is to visualize 3D point cloud of the images captured by camera.

Following (except added ROS path description) is from the original README of DenseDepth.

## ROS

* Run `cd ros`
Then 
* Run `python DenseDepth.py`

## Results

* KITTI
<p align="center"><img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_01.jpg" alt="KITTI"></p>

* NYU Depth V2
<p align="center">
  <img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_02.jpg" alt="NYU Depth v2">
  <img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_03.jpg" alt="NYU Depth v2 table">
</p>

## Requirements
* This code is tested with Keras 2.2.4, Tensorflow 1.13, CUDA 9.0, on a machine with an NVIDIA Titan V and 16GB+ RAM running on Windows 10 or Ubuntu 16.
* Other packages needed `keras pillow matplotlib scikit-learn scikit-image opencv-python pydot` and `GraphViz`.
* Training takes about 20 hours with 4 NVIDIA Titan Xp (or above).

## Pre-trained Models
* [NYU Depth V2](https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5) (165 MB)
* [KITTI](https://s3-eu-west-1.amazonaws.com/densedepth/kitti.h5) (165 MB)

## Demo
* After downloading the pre-trained model (nyu.h5), run `python test.py`. You should see a montage of images with their estimated depth maps.

## Data
* [NYU Depth V2 (50K)](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_data.zip) (4.1 GB): You don't need to extract the dataset since the code loads the entire zip file into memory when training.
* [KITTI](http://www.cvlibs.net/datasets/kitti/): copy the raw data to a folder with the path '../kitti'. Our method expects dense input depth maps, therefore, you need to run a depth [inpainting method](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) on the Lidar data. For our experiments, we used our [Python re-implmentaiton](https://gist.github.com/ialhashim/be6235489a9c43c6d240e8331836586a) of the Matlab code provided with NYU Depth V2 toolbox. The entire 80K images took 2 hours on an 80 nodes cluster for inpainting. For our training, we used the subset defined [here](https://s3-eu-west-1.amazonaws.com/densedepth/kitti_train.csv).
* [Unreal-1k](https://github.com/ialhashim/DenseDepth): coming soon.

## Training
* Run `python train.py --data nyu --gpus 4 --bs 8`.

## Evaluation
* Download, but don't extract, the ground truth test data from [here](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip) (1.4 GB). Then simply run `python evaluate.py`.

## Reference
Corresponding paper to cite:
```
@article{Alhashim2018,
  author    = {Ibraheem Alhashim and Peter Wonka},
  title     = {High Quality Monocular Depth Estimation via Transfer Learning},
  journal   = {arXiv e-prints},
  volume    = {abs/1812.11941},
  year      = {2018},
  url       = {https://arxiv.org/abs/1812.11941},
  eid       = {arXiv:1812.11941},
  eprint    = {1812.11941}
}
```
