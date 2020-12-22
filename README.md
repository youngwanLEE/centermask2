# [CenterMask](https://arxiv.org/abs/1911.06667)2

[[`CenterMask(original code)`](https://github.com/youngwanLEE/CenterMask)][[`vovnet-detectron2`](https://github.com/youngwanLEE/vovnet-detectron2)][[`arxiv`](https://arxiv.org/abs/1911.06667)] [[`BibTeX`](#CitingCenterMask)]

**CenterMask2** is an upgraded implementation on top of [detectron2](https://github.com/facebookresearch/detectron2) beyond original [CenterMask](https://github.com/youngwanLEE/CenterMask) based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

> **[CenterMask : Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667) (CVPR 2020)**<br>
> [Youngwan Lee](https://github.com/youngwanLEE) and Jongyoul Park<br>
> Electronics and Telecommunications Research Institute (ETRI)<br>
> pre-print : https://arxiv.org/abs/1911.06667


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
- Lightweight VoVNet has ben released. (26/02/2020)
- Panoptic-CenterMask has been released. (31/03/2020)
- code update for compatibility with pytorch1.7 and the latest detectron2 (22/12/2020)
## Results on COCO val

### Note

We measure the inference time of all models with batch size 1 on the same V100 GPU machine.

- pytorch1.7.0
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
|**CenterMask2-Lite**|V-19|4x|0.023|**32.8**|**35.9**|<a href="https://dl.dropbox.com/s/dret2ap7djty7mp/centermask2-lite-V-19-eSE-FPN-ms-4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/zsta7azy87a833u/centermask2-lite-V-19-eSE-FPN-ms-4x-metrics.json">metrics</a>
||
|YOLACT550|R-101|4x|0.030|28.2|30.3|[link](https://github.com/dbolya/yolact)
|YOLACT550++|R-50|4x|0.029|34.1|-|[link](https://github.com/dbolya/yolact)
|YOLACT550++|R-101|4x|0.036|34.6|-|[link](https://github.com/dbolya/yolact)
|CenterMask (maskrcnn-benchmark)|V-39|4x|0.027|36.3|40.7|[link](https://github.com/youngwanLEE/CenterMask#coco-val2017-results)
|**CenterMask2-Lite**|V-39|4x|0.028|**36.7**|**40.9**|<a href="https://dl.dropbox.com/s/uwc0ypa1jvco2bi/centermask2-lite-V-39-eSE-FPN-ms-4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/aoa6y3i3el4edbk/centermask2-lite-V-39-eSE-FPN-ms-4x-metrics.json">metrics</a>
* Note that The inference time is measured on Titan Xp GPU for fair comparison with YOLACT.

### Lightweight VoVNet backbone

|Method|Backbone|Param.|lr sched|inference time|mask AP|box AP|download|
|:--------:|:--------:|:--:|:--:|:--:|:----:|:----:|:--------:|
|CenterMask2-Lite|MobileNetV2|3.5M|4x|0.021|27.2|29.8|<a href="https://dl.dropbox.com/s/8omou546f0n78nj/centermask_lite_Mv2_ms_4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/2jlcwy30eq72w47/centermask_lite_Mv2_ms_4x_metrics.json">metrics</a>
||
|CenterMask2-Lite|V-19|11.2M|4x|0.023|32.8|35.9|<a href="https://dl.dropbox.com/s/dret2ap7djty7mp/centermask2-lite-V-19-eSE-FPN-ms-4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/zsta7azy87a833u/centermask2-lite-V-19-eSE-FPN-ms-4x-metrics.json">metrics</a>
|CenterMask2-Lite|V-19-**Slim**|3.1M|4x|0.021|29.8|32.5|<a href="https://dl.dropbox.com/s/o2n1ifl0zkbv16x/centermask-lite-V-19-eSE-slim-FPN-ms-4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/8y71oz0kxwqk7go/centermask-lite-V-19-eSE-slim-FPN-ms-4x-metrics.json?dl=0">metrics</a>
|CenterMask2-Lite|V-19**Slim**-**DW**|1.8M|4x|0.020|27.1|29.5|<a href="https://dl.dropbox.com/s/vsvhwtqm6ko1c7m/centermask-lite-V-19-eSE-slim-dw-FPN-ms-4x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/q4idjnsgvo151zx/centermask-lite-V-19-eSE-slim-dw-FPN-ms-4x-metrics.json">metrics</a>
* _**DW** and **Slim** denote depthwise separable convolution and a thiner model with half the channel size, respectively._   
* __Params.__ means the number of parameters of backbone.   

### Deformable VoVNet Backbone

|Method|Backbone|lr sched|inference time|mask AP|box AP|download|
|:--------:|:--------:|:--:|:--:|:--:|:----:|:----:|
CenterMask2|V2-39|3x|0.050|39.7|44.2|<a href="https://dl.dropbox.com/s/tczecsdxt10uai5/centermask2-V-39-eSE-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/rhoo6vkvh7rjdf9/centermask2-V-39-eSE-FPN-ms-3x_metrics.json">metrics</a>
CenterMask2|V2-39-DCN|3x|0.061|40.3|45.1|<a href="https://dl.dropbox.com/s/zmps03vghzirk7v/centermask-V-39-eSE-dcn-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/aj1mr8m32z11zbw/centermask-V-39-eSE-dcn-FPN-ms-3x-metrics.json">metrics</a>
||
CenterMask2|V2-57|3x|0.058|40.5|45.1|<a href="https://dl.dropbox.com/s/lw8nxajv1tim8gr/centermask2-V-57-eSE-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/x7r5ys3c81ldgq0/centermask2-V-57-eSE-FPN-ms-3x_metrics.json">metrics</a>
CenterMask2|V2-57-DCN|3x|0.071|40.9|45.5|<a href="https://dl.dropbox.com/s/1f64azqyd2ot6qq/centermask-V-57-eSE-dcn-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/b3zpguko137r6eh/centermask-V-57-eSE-dcn-FPN-ms-3x-metrics.json">metrics</a>
||
CenterMask2|V2-99|3x|0.077|41.4|46.0|<a href="https://dl.dropbox.com/s/c6n79x83xkdowqc/centermask2-V-99-eSE-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/jdzgmdatit00hq5/centermask2-V-99-eSE-FPN-ms-3x_metrics.json">metrics</a>
CenterMask2|V2-99-DCN|3x|0.110|42.0|46.9|<a href="https://dl.dropbox.com/s/atuph90nzm7s8x8/centermask-V-99-eSE-dcn-FPN-ms-3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/82ulexlivy19cve/centermask-V-99-eSE-dcn-FPN-ms-3x-metrics.json">metrics</a>
||

* _DCN denotes deformable convolutional networks v2. Note that we apply deformable convolutions from stage 3 to 5 in backbones._

### Panoptic-CenterMask

|Method|Backbone|lr sched|inference time|mask AP|box AP|PQ|download|
|:--------:|:--------:|:--:|:--:|:--:|:----:|:----:|:--------:|
|Panoptic-FPN|R-50|3x|0.063|40.0|36.5|41.5|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/metrics.json">metrics</a>
|Panoptic-CenterMask|R-50|3x|0.063|41.4|37.3|42.0|<a href="https://dl.dropbox.com/s/vxe51cdeprao94j/panoptic_centermask_R_50_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/dfddgx6rnw1zr4l/panoptic_centermask_R_50_ms_3x_metrics.json">metrics</a>
|Panoptic-FPN|V-39|3x|0.063|42.8|38.5|43.4|<a href="https://dl.dropbox.com/s/fnr9r4arv0cbfbf/panoptic_V_39_eSE_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/vftfukrjuu7w1ao/panoptic_V_39_eSE_3x_metrics.json">metrics</a>
|Panoptic-CenterMask|V-39|3x|0.066|43.4|39.0|43.7|<a href="https://dl.dropbox.com/s/49ig16ailra1f4t/panoptic_centermask_V_39_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/wy4mn8n513k0um5/panoptic_centermask_V_39_eSE_ms_3x_metrics.json">metrics</a>
||
|Panoptic-FPN|R-101|3x|0.078|42.4|38.5|43.0|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/metrics.json">metrics</a>
|Panoptic-CenterMask|R-101|3x|0.076|43.5|39.0|43.6|<a href="https://dl.dropbox.com/s/y5stg3qx72gff5o/panoptic_centermask_R_101_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/ojljt0obp8vnr8s/panoptic_centermask_R_101_ms_3x_metrics.json">metrics</a>
|Panoptic-FPN|V-57|3x|0.070|43.4|39.2|44.3|<a href="https://www.dropbox.com/s/zhoqx5rvc0jj0oa/panoptic_V_57_eSE_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/20hwrmru15dilre/panoptic_V_57_eSE_3x_metrics.json">metrics</a>
|Panoptic-CenterMask|V-57|3x|0.071|43.9|39.6|44.5|<a href="https://dl.dropbox.com/s/kqukww4y7tbgbrh/panoptic_centermask_V_57_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/4asto3b4iya74ak/panoptic_centermask_V_57_ms_3x_metrics.json">metrics</a>
||
|Panoptic-CenterMask|V-99|3x|0.091|45.1|40.6|45.4|<a href="https://dl.dropbox.com/s/pr6a3inpasn7qlz/panoptic_centermask_V_99_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/00e8x0riplme7pm/panoptic_centermask_V_99_ms_3x_metrics.json">metrics</a>


## Installation
All you need to use centermask2 is [detectron2](https://github.com/facebookresearch/detectron2). It's easy!    
you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).   
Prepare for coco dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

## Training

#### ImageNet Pretrained Models

We provide backbone weights pretrained on ImageNet-1k dataset for detectron2.
* [MobileNet-V2](https://dl.dropbox.com/s/yduxbc13s3ip6qn/mobilenet_v2_detectron2.pth)
* [VoVNetV2-19-Slim-DW](https://dl.dropbox.com/s/f3s7ospitqoals1/vovnet19_ese_slim_dw_detectron2.pth)
* [VoVNetV2-19-Slim](https://dl.dropbox.com/s/8h5ybmi4ftbcom0/vovnet19_ese_slim_detectron2.pth)
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
 - [x] Adding Lightweight models
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

@inproceedings{lee2020centermask,
  title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
  author={Lee, Youngwan and Park, Jongyoul},
  booktitle={CVPR},
  year={2020}
}
```

## Special Thanks to

[mask scoring for detectron2](https://github.com/lsrock1/maskscoring_rcnn.detectron2) by [Sangrok Lee](https://github.com/lsrock1)   
[FCOS_for_detectron2](https://github.com/aim-uofa/adet) by AdeliDet team.
