#!/bin/sh
# Requierments python3.5 or greater
# Python requirements in requests.txt
python3 Inference2D.py --gpu_id 0 --model_path ./Models/DIC-C2DH-HeLa/ --sequence_path '../DIC-C2DH-HeLa/02' --output_path '../DIC-C2DH-HeLa/02_RES' --edge_dist 30 --edge_thresh 0.5 --min_cell_size 1000 --max_cell_size 10000000 --fov 25 --centers_sigmoid_threshold 0.8 --min_center_size 500 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./DIC_C2DH-HeLa/02



