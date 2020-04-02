# Skeleton-based-action-recognition
Yolov3, Openpose, Tensorflow2, ROS, multi-thread

This is my final year project "3D Action Recognition based on Openpose and YOLO".

## Prerequisite

### 0. install openpose python api

Following the openpose homepage instruction to install [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md) and compile the [python api](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#python-api).

Modify the config.py. Changeing the path about the YOLO data, such as `YOLO.CLASSES` and `YOLO.ANCHORS`.

Change your openpose python api path, so that your code can import pyopenpose correctly.
Additionally, you also have to change the openpose model path in `Module/poser.py`.


### 1. create a conda env.

```
conda create -n tensorflow2 python=3.6
pip install -r requirements.txt
```

### 2. create a ROS package

At first, following the [ros wiki](http://wiki.ros.org/cn/ROS/Tutorials/InstallingandConfiguringROSEnvironment) instruction to install ROS and create a ROS workspace

Then, create a ROS package which names `act_recognizer`.
```
cd catkin_ws/src
mkdir -p act_recognizer
```

### 3. download this repo

Copy all files to ROS package `act_recognizer`

### 4. download yolo and mlp checkpoints

Download checkpoints from [BaiduYun](www). Then move `yolov3.weights` into checkpoints folder `checkpoints/YOLO` and  `mlp.h5` to  `checkpoints`.

```
cd act_recognizer/src
mkdir -p checkpoints/YOLO
```

### 5. run the code

The ROS package whic written by Python do not need to compile. 
We have to specify the python interpreter in all `.py` files in the first line to use the conda env `tensorflow2`'s python interpreter.
Likes this,

```
#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
```

Then, run the code following,
```
roscore
cd catkin_ws
conda activate tensorflow2
source ./devel/setup.bash
roslaunch act_recognizer run.launch
```

## Citation and Reference
[Openpose from CMU](https://github.com/kevinchan04/openpose)

[Yolov3 tensorflow2 from YunYang1994](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3)
