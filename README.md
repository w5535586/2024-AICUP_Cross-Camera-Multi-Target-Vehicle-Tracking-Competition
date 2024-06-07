# 2024-AICUP_Cross-Camera-Multi-Target-Vehicle-Tracking-Competition: TEAM_5013
Source Code是對AICUP舉辦的跨相機多目標車輛追蹤競賽
下面會有從建立環境、訓練模型、跟檢測的操作流程與步驟。

## Our training conditions
CPU: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz  
GPU: 2080Ti * 2  
RAM: 64GB  
python 3.10.14  
pytorch 2.1.0

## Install requirements for implement
```
pip install -r requirements.txt
```
## Prepare model weight
請下載[TAPNET weight](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt)放入checkpoints內

## Training

Data preparation

```
請採用YOLO V9的官方資料格式
```

Training YOLO V9 model

``` 
python train_dual.py --workers 8 --device 1 --batch 2 --data /ssd1/hai/aicup2024/AICUP_Baseline_BoT-SORT-main/yolov9/data/AICUP.yaml --img 1280 --cfg models/detect/yolov9-e.yaml --weights '' --name yolov9-e --hyp hyp.scratch-high.yaml --min-items 0 --epochs 10 --close-mosaic 15
```

## Inference (tracking)
單資料夾追蹤:
``` 
 python mc_demo_yolov9.py --weights "Your model weight path" --source "Your data folder path" --device "0,1" --name "your output folder name" --fuse-score --agnostic-nms
```

競賽結果輸出:
``` 
 bash track_all_timestamps_v9.sh --weights "Your model weight path" --source-dir "Your data folder path" --device "0"
```
輸出的結果檔案會在mc_demo_yolov9.py的設定
```
with open("Your folder Path"+  f"{opt.name}.txt", 'w') as f:
        f.writelines(results)
``` 
## Reference 
This code is based on [yolov9](https://github.com/WongKinYiu/yolov9)  
This code is based on [tapenet](https://github.com/google-deepmind/tapnet)
