#!/bin/sh
# Requierments python3.5
python3 Inference2D.py --gpu_id 0 --model_path './Models/Fluo-C2DL-MSC/' --sequence_path '../Fluo-C2DL-MSC/01' --output_path '../Fluo-C2DL-MSC/01_RES' --edge_dist 10 --edge_thresh 0.3 --min_cell_size 1000 --max_cell_size 10000000 --fov 25 --centers_sigmoid_threshold 0.4 --min_center_size 500 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-C2DL-MSC/01



