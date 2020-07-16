#!/bin/sh
# Requierments python3.5 or greater
# Python requirements in requests.txt
python3 Inference2D.py --gpu_id 0 --model_path ./Models/PhC-C2DH-U373/ --sequence_path '../PhC-C2DH-U373/02' --output_path '../PhC-C2DH-U373/02_RES' --edge_dist 20 --edge_thresh 0.3 --min_cell_size 1000 --max_cell_size 100000 --fov 0 --centers_sigmoid_threshold 1 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./PhC-C2DH-U373/02



