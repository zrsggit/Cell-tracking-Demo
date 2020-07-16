#!/bin/sh
# Requierments python3.5
python3 Inference2D.py --gpu_id 0 --model_path './Models/Fluo-N2DH-GOWT1/' --sequence_path '../Fluo-N2DH-GOWT1/01' --output_path '../Fluo-N2DH-GOWT1/01_RES' --edge_dist 2 --min_cell_size 10 --edge_thresh 0.33 --max_cell_size 1000000 --fov 10 --centers_sigmoid_threshold 0.3 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-N2DH-GOWT1/01



