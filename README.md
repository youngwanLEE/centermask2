# [CenterMask](https://arxiv.org/abs/1911.06667)2

[[`CenterMask(original code)`](https://github.com/youngwanLEE/CenterMask)][[`vovnet-detectron2`](https://github.com/youngwanLEE/vovnet-detectron2)][[`arxiv`](https://arxiv.org/abs/1911.06667)] [[`BibTeX`](#CitingCenterMask)]

**CenterMask2** is an upgraded implementation on top of [detectron2](https://github.com/facebookresearch/detectron2) beyond original [CenterMask](https://github.com/youngwanLEE/CenterMask) based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).


<div align="center">
  <img src="https://dl.dropbox.com/s/yg9zr1tvljoeuyi/architecture.png" width="850px" />
</div>

  
  

## Highlights
- ***First* anchor-free one-stage instance segmentation.** To the best of our knowledge, **CenterMask** is the first instance segmentation on top of anchor-free object detection (15/11/2019).
- **Toward Real-Time: CenterMask-Lite.**  This works provide not only large-scale CenterMask but also lightweight CenterMask-Lite that can run at real-time speed (> 30 fps).
- **State-of-the-art performance.**  CenterMask outperforms Mask R-CNN, TensorMask, and ShapeMask at much faster speed and CenterMask-Lite models also surpass YOLACT or YOLACT++ by large margins.
- **Well balanced (speed/accuracy) backbone network, VoVNetV2.**  VoVNetV2 shows better performance and faster speed than ResNe(X)t or HRNet.


## Updates
- CenterMask2 has been released. (20/02/2020)

## Results on COCO val

### Note

We measure the inference time of all models with batch size 1 on the same V100 GPU machine.

- pytorch1.3.1
- CUDA 10.1
- cuDNN 7.3
- multi-scale augmentation
- Unless speficified, no Test-Time Augmentation (TTA)



### CenterMask

|Method|Backbone|lr sched|inference time|mask AP|box AP|download|
|:--------:|:--------:|:--:|:--:|----|----|:--------:|
Mask R-CNN (detectron2)|R-50|3x|0.055|37.2|41.0|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a>
Mask R-CNN (detectron2)|V2-39|3x|0.052|39.3|43.8|<a href="https://dl.dropbox.com/s/dkto39ececze6l4/faster_V_39_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/dx9qz1dn65ccrwd/faster_V_39_eSE_ms_3x_metrics.json">metrics</a>
CenterMask (maskrcnn-benchmark)|V2-39|3x|0.070|38.5|43.5|[link](https://github.com/youngwanLEE/CenterMask#coco-val2017-results)
**CenterMask2**|V2-39|3x|**0.050**|**39.7**|**44.2**|<a href="https://dl.dropbox.com/s/tczecsdxt10uai5/centermask2-V-39-eSE-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/rhoo6vkvh7rjdf9/centermask2-V-39-eSE-FPN-ms-3x_metrics.json">metrics</a>
||
Mask R-CNN (detectron2)|R-101|3x|0.070|38.6|42.9|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/metrics.json">metrics</a>
Mask R-CNN (detectron2)|V2-57|3x|0.058|39.7|44.2|<a href="https://dl.dropbox.com/s/c7mb1mq10eo4pzk/faster_V_57_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/3tsn218zzmuhyo8/faster_V_57_eSE_metrics.json">metrics</a>
CenterMask (maskrcnn-benchmark)|V2-57|3x|0.076|39.4|44.6|[link](https://github.com/youngwanLEE/CenterMask#coco-val2017-results)
**CenterMask2**|V2-57|3x|**0.058**|**40.5**|**45.1**|<a href="https://dl.dropbox.com/s/lw8nxajv1tim8gr/centermask2-V-57-eSE-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/x7r5ys3c81ldgq0/centermask2-V-57-eSE-FPN-ms-3x_metrics.json">metrics</a>
||
Mask R-CNN (detectron2)|X-101|3x|0.129|39.5|44.3|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/metrics.json">metrics</a>
Mask R-CNN (detectron2)|V2-99|3x|0.076|40.3|44.9|<a href="https://dl.dropbox.com/s/v64mknwzfpmfcdh/faster_V_99_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/zvaz9s8gvq2mhrd/faster_V_99_eSE_ms_3x_metrics.json">metrics</a>
CenterMask (maskrcnn-benchmark)|V2-99|3x|0.106|40.2|45.6|[link](https://github.com/youngwanLEE/CenterMask#coco-val2017-results)
**CenterMask2**|V2-99|3x|**0.077**|**41.4**|**46.0**|<a href="https://dl.dropbox.com/s/c6n79x83xkdowqc/centermask2-V-99-eSE-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/jdzgmdatit00hq5/centermask2-V-99-eSE-FPN-ms-3x_metrics.json">metrics</a>
||
**CenterMask2 (TTA)**|V2-99|3x|-|**42.5**|**48.6**|<a href="https://dl.dropbox.com/s/c6n79x83xkdowqc/centermask2-V-99-eSE-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/jdzgmdatit00hq5/centermask2-V-99-eSE-FPN-ms-3x_metrics.json">metrics</a>
* TTA denotes Test-Time Augmentation (multi-scale test).

### CenterMask-Lite

|Method|Backbone|lr sched|inference time|mask AP|box AP|download|
|:--------:|:--------:|:--:|:--:|:----:|:----:|:--------:|
|YOLACT550|R-50|4x|0.023|28.2|30.3|[link](https://github.com/dbolya/yolact)
|CenterMask (maskrcnn-benchmark)|V-19|4x|0.023|32.4|35.9|[link](https://github.com/youngwanLEE/CenterMask#coco-val2017-results)
|**CenterMask2**|V-19|4x|0.023|**32.8**|**35.9**|<a href="https://dl.dropbox.com/s/dret2ap7djty7mp/centermask2-lite-V-19-eSE-FPN-ms-4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/zsta7azy87a833u/centermask2-lite-V-19-eSE-FPN-ms-4x-metrics.json">metrics</a>
||
|YOLACT550|R-101|4x|0.030|28.2|30.3|[link](https://github.com/dbolya/yolact)
|YOLACT550++|R-50|4x|0.029|34.1|-|[link](https://github.com/dbolya/yolact)
|YOLACT550++|R-101|4x|0.036|34.6|-|[link](https://github.com/dbolya/yolact)
|CenterMask (maskrcnn-benchmark)|V-39|4x|0.027|36.3|40.7|[link](https://github.com/youngwanLEE/CenterMask#coco-val2017-results)
|**CenterMask2**|V-39|4x|0.028|**36.7**|**40.9**|<a href="https://dl.dropbox.com/s/uwc0ypa1jvco2bi/centermask2-lite-V-39-eSE-FPN-ms-4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/aoa6y3i3el4edbk/centermask2-lite-V-39-eSE-FPN-ms-4x-metrics.json">metrics</a>
* Note that The inference time is measured on Titan Xp GPU for fair comparison with YOLACT.



## Installation
All you need to use centermask2 is [detectron2](https://github.com/facebookresearch/detectron2). It's easy!    
you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).   
Prepare for coco dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

## Training

#### ImageNet Pretrained Models

We provide backbone weights pretrained on ImageNet-1k dataset.
* [VoVNetV2-19](https://dl.dropbox.com/s/rptgw6stppbiw1u/vovnet19_ese_detectron2.pth)
* [VoVNetV2-39](https://dl.dropbox.com/s/q98pypf96rhtd8y/vovnet39_ese_detectron2.pth)
* [VoVNetV2-57](https://dl.dropbox.com/s/8xl0cb3jj51f45a/vovnet57_ese_detectron2.pth)
* [VoVNetV2-99](https://dl.dropbox.com/s/1mlv31coewx8trd/vovnet99_ese_detectron2.pth)


To train a model, run
```bash
cd centermask2
python train_net.py --config-file "configs/<config.yaml>"
```

For example, to launch CenterMask training with VoVNetV2-39 backbone on 8 GPUs,
one should execute:
```bash
cd centermask2
python train_net.py --config-file "configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml" --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:   
* if you want to inference with 1 batch `--num-gpus 1` 
* `--eval-only`
* `MODEL.WEIGHTS path/to/the/model.pth`

```bash
cd centermask2
wget https://dl.dropbox.com/s/tczecsdxt10uai5/centermask2-V-39-eSE-FPN-ms-3x.pth
python train_net.py --config-file "configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml" --num-gpus 1 --eval-only MODEL.WEIGHTS centermask2-V-39-eSE-FPN-ms-3x.pth
```

## TODO
 - [ ] Adding Lightweight models
 - [ ] Applying CenterMask for PointRend or Panoptic-FPN.


## <a name="CitingCenterMask"></a>Citing CenterMask

If you use VoVNet, please use the following BibTeX entry.

```BibTeX
@inproceedings{lee2019energy,
  title = {An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection},
  author = {Lee, Youngwan and Hwang, Joong-won and Lee, Sangrok and Bae, Yuseok and Park, Jongyoul},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2019}
}

@article{lee2019centermask,
  title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
  author={Lee, Youngwan and Park, Jongyoul},
  journal={arXiv preprint arXiv:1911.06667},
  year={2019}
}
```

## Special Thanks to

[mask scoring for detectron2](https://github.com/lsrock1/maskscoring_rcnn.detectron2) by [Sangrok Lee](https://github.com/lsrock1)   
[FCOS_for_detectron2](https://github.com/aim-uofa/adet) by AdeliDet team.