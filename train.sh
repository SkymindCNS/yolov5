#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"

pip install -r requirements.txt
apt-get install ffmpeg libsm6 libxext6  -y

# debug
python train_from_pth_dataset.py --img 640 --batch 8 --epochs 60 --data data/testkits.yaml --weights yolov5s.pt --device 0 --workers 0

#model m
# python train_from_pth_dataset.py --img 640 --batch 32 --epochs 60 --data data/testkits.yaml --weights yolov5s.pt --device 0 --workers 0

#model s
# python data/shared/vpass-experiments/yolov5/train.py --img 640 --batch 80 --epochs 30 --data data/shared/vpass-experiments/yolov5/data/testkits.yaml --weights yolov5s.pt --device 0 --workers 0

#transfer learning