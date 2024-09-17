<p align="center">
  <h1 align="center">MA-Stereo: Real-Time Stereo Matching via Multi-scale Attention Fusion and Spatial Error-aware Refinement
</h1>
  <p align="center">
    Wei Gao, Yongjie Cai, Youssef Akoudad
  </p>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/1434205229/MA-Stereo/blob/main/image/MA-Stereo.png" alt="Logo" width="90%">
  </a>
</p>

# How to use

## Data Preparation
* [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)
  
## Train
Use the following command to train MA-Stereo on Scene Flow.
First training,
```
python train_sceneflow.py --logdir ./checkpoints/sceneflow/first/
```
Second training,
```
python train_sceneflow_2.py --logdir ./checkpoints/sceneflow/second/ --loadckpt ./checkpoints/sceneflow/first/checkpoint_000063.ckpt
```

Use the following command to train CGI-Stereo on KITTI (using pretrained model on Scene Flow),
```
python train_kitti.py --logdir ./checkpoints/kitti/ --loadckpt ./checkpoints/sceneflow/second/checkpoint_000063.ckpt
```

## Evaluation on Scene Flow and KITTI

### Pretrained Model
* [Google Drive](https://drive.google.com/drive/folders/1f9aIJpSsOPgMczmvFuN0f3_Ui775tvjV?usp=drive_link)

Generate disparity images of KITTI test set,
```
python save_disp.py
```


# Acknowledgement

Special thanks to CGI-Stereo for providing the code base for this work.

<details>
<summary>
<a href="https://github.com/gangweiX/CGI-Stereo">CGI-Stereo</a> 
</summary>

