#!/bin/sh
# Requierments python3.5 or greater
# Python requirements in requests.txt
python3 Inference2D.py --gpu_id 0 --model_path ./Models/PhC-C2DL-PSC/ --sequence_path '../PhC-C2DL-PSC/01' --output_path '../PhC-C2DL-PSC/01_RES' --edge_dist  3 --edge_thresh 0.3  --min_cell_size 5 --max_cell_size 1000 --fov 10 --centers_sigmoid_threshold 0.6 --min_center_size 1 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./PhC-C2DL-PSC/01



