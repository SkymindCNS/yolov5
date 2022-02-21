#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"

pip install -r requirements.txt
apt-get install ffmpeg libsm6 libxext6  -y

python train_from_pth_dataset.py