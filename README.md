
# How to use

## Environment
* Python 3.8
* Pytorch 1.10

## Install

### Create a virtual environment and activate it.

```
conda create -n acvnet python=3.8
conda activate acvnet
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboardX
pip install matplotlib 
pip install tqdm
```

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train
Use the following command to train VFDSNet on Scene Flow

Firstly, train attention volume generation network for 64 epochs,
```
python main.py --attention_weights_only True
```
Secondly, freeze attention volume generation network parameters, train the remaining network for another 64 epochs,
```
python main.py --freeze_attention_weights True
``````

## Test
```
python test_kitti_sceneflow.py
```



# VFDSNet
